import numpy as np
import time
from scipy.special import erfinv

def _lorenz96_rhs(x, F):
    """Compute the Lorenz-96 RHS for a single state vector."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        ip1 = (i + 1) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx

def _lorenz96_rhs_batch(X, F):
    """Compute Lorenz-96 RHS for a batch of states (n_members, n_vars)."""
    n_vars = X.shape[1]
    # Use rolling for efficiency
    x_im2 = np.roll(X, 2, axis=1)
    x_im1 = np.roll(X, 1, axis=1)
    x_ip1 = np.roll(X, -1, axis=1)
    return (x_ip1 - x_im2) * x_im1 - X + F

def _rk4_step(X, F, dt):
    """Advance ensemble by one RK4 step."""
    k1 = _lorenz96_rhs_batch(X, F)
    k2 = _lorenz96_rhs_batch(X + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs_batch(X + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs_batch(X + dt * k3, F)
    return X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _gaspari_cohn(d, L):
    """Gaspari-Cohn correlation function."""
    d = np.abs(d)
    L = max(L, 1e-8)
    out = np.zeros_like(d)
    mask1 = d <= L
    mask2 = (d > L) & (d <= 2 * L)
    dl = d[mask1]
    out[mask1] = (-5.0 / 4.0 * dl**5 / L**5 + 5.0 * dl**4 / L**4 - 10.0 / 3.0 * dl**3 / L**3 + 1.0)
    dr = d[mask2] - L
    out[mask2] = (1.0 / 12.0 * dr**5 / L**5 - 0.5 * dr**4 / L**4 + dr**3 / L**3 - 5.0 / 3.0 * dr**2 / L**2 + 1.0 + 4.0 / 3.0 * dr / L)
    return out

def _ring_distance(i, j, n):
    """Minimum ring distance between indices i and j on a ring of size n."""
    d = np.abs(i - j)
    return np.minimum(d, n - d)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using a localized ensemble Kalman filter
    with adaptive inflation and deterministic perturbed observations.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Infer forcing F from the ensemble spread/mean characteristics
    # For Lorenz-96, F=6,8,10 are typical. We'll use a heuristic.
    # Actually, we don't have F directly. But we can estimate it.
    # The equilibrium mean is roughly F^2 for each variable.
    # Let's estimate F from the mean of the ensemble.
    mean_x = np.mean(ensemble, axis=0)
    # For L96, <x_i> ≈ F^2 for large N
    F_est = np.sqrt(np.mean(mean_x))
    # Clamp to reasonable range
    F_est = np.clip(F_est, 4.0, 12.0)
    
    # Observation noise std
    obs_std = np.sqrt(observation_variance)
    
    # Adaptive inflation parameter
    inflation = 1.02  # Start with mild inflation
    
    result = ensemble.copy()
    
    # Localize observations
    # Gaspari-Cohn localization length
    L_loc = 5  # In grid units
    
    # Build localization matrix for observed variables
    # For each observed index, we need to correlate with all variables
    # This is O(n_obs * n_vars) which is manageable for n_vars=20
    
    # Perturb observations deterministically to avoid rank deficiency
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    obs_perturbed = observations + rng.randn(n_members, n_obs) * obs_std
    
    # For each observed variable, perform a localized EnKF update
    # We process them sequentially with localization
    
    # Precompute ring distances
    all_indices = np.arange(n_vars)
    dist_matrix = np.zeros((n_obs, n_vars))
    for k, obs_idx in enumerate(observed_indices):
        dist_matrix[k, :] = _ring_distance(obs_idx, all_indices, n_vars)
    
    # Localization weights
    loc_weights = _gaspari_cohn(dist_matrix, L_loc)
    # Ensure the observed variable itself has weight 1
    for k in range(n_obs):
        loc_weights[k, observed_indices[k]] = 1.0
    
    # Update ensemble
    for k in range(n_obs):
        obs_idx = observed_indices[k]
        w = loc_weights[k, :]
        
        # Check deadline
        if time.time() > deadline - 1.0:
            break
        
        # Localize the update
        # The innovation is (obs_perturbed[:, k] - ensemble[:, obs_idx])
        innovation = obs_perturbed[:, k] - result[:, obs_idx]
        
        # Compute ensemble mean and covariance of the observed variable
        obs_mean = np.mean(result[:, obs_idx])
        obs_var = np.var(result[:, obs_idx], ddof=1)
        
        # Avoid division by zero
        obs_var = max(obs_var, 1e-6)
        
        # Kalman gain for the observed variable
        K_obs = obs_var / (obs_var + observation_variance)
        
        # Update the observed variable
        result[:, obs_idx] += K_obs * innovation
        
        # Update unobserved variables using cross-covariance
        for j in range(n_vars):
            if j == obs_idx:
                continue
            if w[j] < 1e-6:
                continue
            
            # Cross-covariance between obs_idx and j
            cross_cov = np.cov(result[:, obs_idx], result[:, j], ddof=1)
            var_j = np.var(result[:, j], ddof=1)
            var_j = max(var_j, 1e-6)
            
            K_j = w[j] * cross_cov / (obs_var + observation_variance)
            
            result[:, j] += K_j * innovation
    
    # Adaptive inflation: adjust ensemble spread
    # Compare ensemble variance to observation variance at observed points
    ens_var_obs = np.var(result[:, observed_indices], axis=0, ddof=1)
    mean_ens_var = np.mean(ens_var_obs)
    target_var = observation_variance + mean_ens_var * 0.5
    
    # Adjust inflation
    if mean_ens_var > 0:
        inflation_factor = np.sqrt(target_var / mean_ens_var)
        inflation_factor = np.clip(inflation_factor, 0.95, 1.10)
        
        ens_mean = np.mean(result, axis=0)
        result = ens_mean + (result - ens_mean) * inflation_factor
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
