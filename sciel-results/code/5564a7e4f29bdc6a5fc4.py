"""Lorenz-96 partial-observation ensemble assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS for state x with forcing F."""
    n = len(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _propagate_ensemble(ensemble, F, dt, n_steps):
    """Propagate each ensemble member forward by n_steps RK4 steps."""
    result = ensemble.copy()
    for i in range(n_steps):
        for m in range(len(result)):
            result[m] = _rk4_step(result[m], F, dt)
    return result

def _infer_F(ensemble, observations, observed_indices):
    """
    Infer the forcing parameter F by matching the mean dynamics.
    We use the fact that in Lorenz-96, the mean of (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F = 0
    in the steady state. So F ≈ mean of x_i - (x_{i+1}-x_{i-2})*x_{i-1}.
    But we only have observations of half the coordinates. We can estimate F from
    the ensemble mean using the observed coordinates and assuming the unobserved
    ones are close to the ensemble mean.
    
    Actually, a simpler approach: use the relation that in the attractor,
    the average of x_i is related to F. For Lorenz-96, the mean state satisfies
    F ≈ mean(x) approximately for large N, but more precisely we can use:
    F = mean over i of [x_i - (x_{i+1} - x_{i-2})*x_{i-1}]
    
    Since we only observe even indices, we'll use those and interpolate.
    """
    n = len(ensemble[0])
    mean_state = np.mean(ensemble, axis=0)
    
    # Use observed indices to estimate F
    # For each observed index i, we need x_{i+1}, x_{i-2}, x_{i-1}
    # We'll use the ensemble mean for all components
    F_estimates = []
    for i in observed_indices:
        ip1 = (i + 1) % n
        im2 = (i - 2) % n
        im1 = (i - 1) % n
        # Using ensemble mean as proxy
        val = mean_state[i] - (mean_state[ip1] - mean_state[im2]) * mean_state[im1]
        F_estimates.append(val)
    
    F = np.mean(F_estimates)
    # Clamp to reasonable range
    F = np.clip(F, 1.0, 15.0)
    return F

def _localization_kernel(distances, length_scale):
    """Gaussian localization kernel."""
    return np.exp(-0.5 * (distances / length_scale) ** 2)

def _ensemble_update(ensemble, observations, observed_indices, observation_variance, F):
    """
    Perform an ensemble Kalman filter update with localization.
    """
    n_members = len(ensemble)
    n_vars = len(ensemble[0])
    n_obs = len(observed_indices)
    
    # Ensemble mean and perturbed observations
    mean_state = np.mean(ensemble, axis=0)
    obs_mean = np.mean(observations, axis=0)
    
    # Anomalies
    X = ensemble - mean_state  # (n_members, n_vars)
    Y = observations - obs_mean  # (n_members, n_obs)
    
    # Covariances
    # P_XX = X^T X / (n_members - 1)  (n_vars x n_vars)
    # P_XY = X^T Y / (n_members - 1)  (n_vars x n_obs)
    # P_YY = Y^T Y / (n_members - 1) + R  (n_obs x n_obs)
    
    # Use the ensemble-based formulation to avoid forming large matrices
    # K = P_XY @ inv(P_YY)
    # We'll use the SVD-based approach for numerical stability
    
    # P_YY = Y^T Y / (n-1) + R
    P_YY = (Y.T @ Y) / (n_members - 1.0)
    P_YY += observation_variance * np.eye(n_obs)
    
    # P_XY = X^T Y / (n-1)
    P_XY = (X.T @ Y) / (n_members - 1.0)
    
    # Apply localization: for each state variable i, scale the observation influence
    # based on distance from observed indices
    # We'll apply localization to the gain matrix
    # First compute the gain K = P_XY @ inv(P_YY)
    try:
        K = P_XY @ np.linalg.inv(P_YY)  # (n_vars x n_obs)
    except np.linalg.LinAlgError:
        K = P_XY @ np.linalg.pinv(P_YY)
    
    # Apply localization
    # For each state variable i, compute the weighted sum of observation influences
    # based on distance to each observed index
    length_scale = 4.0  # localization length scale
    n = n_vars
    
    # Build localization weights: for each state var i and obs var j, weight based on distance
    # dist(i, j) = min(|i-j|, n - |i-j|)
    state_idx = np.arange(n)
    obs_idx = np.array(observed_indices)
    
    # distances[i, j] = circular distance between state_idx[i] and obs_idx[j]
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, n - diffs)
    
    # Localization weights
    W = _localization_kernel(distances, length_scale)  # (n_vars x n_obs)
    
    # Apply localization to gain: K_loc[i, j] = K[i, j] * W[i, j]
    K_loc = K * W
    
    # Update: x_new = x + K_loc @ (y - H x)
    # H x = x[observed_indices]
    innovation = observations - ensemble[:, observed_indices]  # (n_members, n_obs)
    
    updated = ensemble.copy()
    for m in range(n_members):
        update = K_loc @ innovation[m]  # (n_vars,)
        updated[m] = ensemble[m] + update
    
    return updated

def _inflation(ensemble, mean_state, observations, observed_indices, observation_variance):
    """
    Simple inflation to prevent filter divergence.
    """
    # Estimate the observation-space variance
    obs_pred = np.mean(ensemble[:, observed_indices], axis=0)
    obs_pred_var = np.var(ensemble[:, observed_indices], axis=0)
    
    # Inflate if the ensemble variance is too small relative to observation variance
    target_var = observation_variance
    min_var = 0.5 * target_var
    
    # Compute inflation factor
    ratios = np.maximum(obs_pred_var, 1e-10) / np.maximum(target_var, 1e-10)
    # If ratio < 1, inflate
    inflation_factor = np.sqrt(np.maximum(1.0, 1.0 / ratios))
    inflation_factor = np.clip(inflation_factor, 1.0, 2.0)
    
    # Apply inflation to observed components
    for m in range(len(ensemble)):
        for j, idx in enumerate(observed_indices):
            ensemble[m, idx] = mean_state[idx] + inflation_factor[j] * (ensemble[m, idx] - mean_state[idx])
    
    return ensemble

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 with partial observations.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each with 20 state variables.
    observations : ndarray, shape (16, 10)
        Noisy observations of the 10 even-indexed variables for each member.
    observed_indices : array-like
        Indices of observed variables (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Infer the forcing parameter F
    F = _infer_F(ensemble, observations, observed_indices)
    
    # Perform ensemble Kalman filter update with localization
    updated = _ensemble_update(ensemble, observations, observed_indices, observation_variance, F)
    
    # Apply mild inflation to prevent filter collapse
    mean_state = np.mean(updated, axis=0)
    updated = _inflation(updated, mean_state, observations, observed_indices, observation_variance)
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have
        pass
    
    # Ensure output is finite and within bounds
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
