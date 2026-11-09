"""Lorenz-96 ensemble assimilation with NLESRF, adaptive inflation, and global localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Non-Local Ensemble Square Root Filter (NLESRF) analysis update for Lorenz-96.
    Uses global Gaussian localization, adaptive multiplicative inflation,
    and a simultaneous update of all 20 variables to preserve covariance structure.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 105.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # --- Adaptive Multiplicative Inflation ---
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target spread: balance between observation noise and model uncertainty
    target_spread = np.sqrt(R_val) * 1.35
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-10))
    
    # Innovation-based adaptive inflation: compare observation residuals to expected
    # Expected innovation variance = ensemble variance at obs + obs noise variance
    ens_var_obs_arr = ens_var_obs
    expected_innov_var = ens_var_obs_arr + R_val
    actual_innov = obs - ens_mean[obs_idx]
    actual_innov_var = np.mean(actual_innov ** 2)
    expected_innov_var_mean = np.mean(expected_innov_var)
    
    # Ratio-based inflation
    ratio_spread = target_spread / np.maximum(current_spread, 1e-10)
    spread_inflation = np.clip(np.mean(ratio_spread), 0.95, 1.25)
    
    # Innovation-based inflation
    innov_ratio = actual_innov_var / np.maximum(expected_innov_var_mean, 1e-10)
    innov_inflation = np.clip(innov_ratio, 0.95, 1.25)
    
    # Blend: 0.6 weight on innovation-based, 0.4 on spread-based
    inflation = 0.6 * innov_inflation + 0.4 * spread_inflation
    inflation = np.clip(inflation, 1.0, 1.18)
    
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    
    # --- Global Gaussian Localization ---
    # Tighter sigma=2.2 to reduce spurious long-range correlations
    sigma = 2.2
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (n_vars, n_obs)
    
    # --- NLESRF: Simultaneous Update of All Variables ---
    # Step 1: Shift observations to zero mean
    obs_shifted = obs - ens_mean[obs_idx]
    
    # Step 2: Anomalies at observed locations
    A_obs = A[:, obs_idx]  # Shape (n_members, n_obs)
    
    # Compute cross-covariance with localization
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # Shape (n_vars, n_obs)
    C_xo_loc = C_xo * loc_weights
    
    # Observation covariance
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # Shape (n_obs, n_obs)
    S = C_oo + R_val * np.eye(n_obs)
    
    # Compute gain using Cholesky for stability
    try:
        L = np.linalg.cholesky(S + 1e-12 * np.eye(n_obs))
        # K = C_xo_loc @ inv(S)
        X = np.linalg.solve(L.T, C_xo_loc.T)
        Y = np.linalg.solve(L, X)
        K = Y.T  # Shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # Update mean
    delta_mean = K @ obs_shifted
    ens_mean_new = ens_mean + delta_mean
    
    # Update anomalies using square-root formulation
    # A_new = A - A_obs @ K^T
    A_obs_Kt = A_obs @ K.T  # Shape (n_members, n_vars)
    A_new = A - A_obs_Kt
    
    # Reconstruct ensemble
    ens = ens_mean_new[None, :] + A_new
    
    # Clip and check for finiteness
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
