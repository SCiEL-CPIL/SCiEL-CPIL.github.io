"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization.
    d: distance matrix (N, N)
    c: correlation length
    """
    d = np.abs(d)
    result = np.zeros_like(d)
    
    # d <= 0.5c
    mask1 = d <= 0.5 * c
    d1 = d[mask1]
    result[mask1] = (-5.0/4.0) * (d1**5 / c**5) + 5.0 * (d1**4 / c**4) - 10.0 * (d1**3 / c**3) + 5.0 * d1
    
    # 0.5c < d <= 1.5c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    d2 = d[mask2]
    result[mask2] = (1.0/12.0) * (d2**5 / c**5) - (5.0/4.0) * (d2**4 / c**4) + (5.0/2.0) * (d2**3 / c**3) - 5.0 * (d2**2 / c**2) + (25.0/12.0) * (d2 / c) + 1.0/2.0
    
    # 1.5c < d <= 2.5c
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    d3 = d[mask3]
    result[mask3] = (-1.0/16.0) * (d3**5 / c**5) + (1.0/4.0) * (d3**4 / c**4) - (15.0/8.0) * (d3**3 / c**3) + (5.0/2.0) * (d3**2 / c**2) - (25.0/12.0) * (d3 / c) + 1.0/4.0
    
    return result


def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    n = len(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def _rk4_step(x, F, dt=0.01, n_steps=5):
    """Propagate Lorenz-96 state with RK4."""
    for _ in range(n_steps):
        k1 = _lorenz96_rhs(x, F)
        k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_rhs(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of model states
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates
    observed_indices : array-like
        Indices of observed coordinates (even indices)
    observation_variance : float
        Variance of observation noise
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = ensemble.shape[0]  # ensemble size
    dim = ensemble.shape[1]  # state dimension
    
    result = ensemble.copy()
    
    # Estimate F from the ensemble mean (rough estimate)
    x_mean = np.mean(ensemble, axis=0)
    
    # Estimate F: in steady state, F ≈ mean of x_i - (x_{i+1}-x_{i-2})*x_{i-1}
    # Actually, we can estimate F from the dynamics
    rhs_vals = np.array([_lorenz96_rhs(x_mean, F=0) for _ in range(1)])
    # F ≈ mean(x) - mean(rhs without F)
    # rhs = (x_{i+1}-x_{i-2})*x_{i-1} - x_i, so F = x_i - (x_{i+1}-x_{i-2})*x_{i-1} averaged
    F_est = np.mean(x_mean - (np.roll(x_mean, -1) - np.roll(x_mean, 2)) * np.roll(x_mean, 1))
    
    # Use a reasonable F estimate, clamp to [6, 10]
    F_est = np.clip(F_est, 6.0, 10.0)
    
    # Observation indices
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Innovation: observation minus ensemble mean at observed locations
    innovation = observations - x_mean[obs_idx]
    
    # Compute ensemble covariance
    anomalies = ensemble - x_mean  # (N, dim)
    
    # Sample covariance
    cov = (anomalies.T @ anomalies) / (N - 1)  # (dim, dim)
    
    # Localization: Gaspari-Cohn with correlation length
    # For Lorenz-96 with 20 variables, use periodic distance
    indices = np.arange(dim)
    dist = np.abs(indices[:, None] - indices[None, :])
    dist = np.minimum(dist, dim - dist)  # periodic distance
    
    # Correlation length in index space
    c = 3.0  # correlation length
    
    # Apply localization to covariance
    loc_matrix = _gaspari_cohn(dist, c)
    cov_loc = cov * loc_matrix
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance: H P H^T + R
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    S = H @ cov_loc @ H.T + R  # (n_obs, n_obs)
    
    # Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = cov_loc @ H.T @ S_inv  # (dim, n_obs)
    
    # Update ensemble
    # x_a = x + K (y - H x)
    for m in range(N):
        result[m] = ensemble[m] + K @ innovation
    
    # Inflate ensemble to prevent collapse
    # Adaptive inflation based on ensemble spread
    spread = np.std(ensemble, axis=0)
    mean_spread = np.mean(spread)
    
    # Simple adaptive inflation
    inflation_factor = 1.005  # slight inflation
    
    result = x_mean + inflation_factor * (result - x_mean)
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        pass
    
    # Ensure finite values and within bounds
    result = np.where(np.isfinite(result), result, x_mean)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
