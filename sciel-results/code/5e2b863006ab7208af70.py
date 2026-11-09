"""Lorenz-96 ensemble assimilation with EnKF, Adaptive Inflation, and Localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) analysis update for Lorenz-96.
    
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
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and centered anomalies
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # 2. Compute covariances BEFORE inflation for innovation-based inflation
    A_pre = ens_centered  # Shape (16, 20)
    A_obs_pre = A_pre[:, obs_idx]  # Shape (16, 10)
    
    # Cross-covariance: Shape (20, 10)
    C_xo_pre = (A_pre.T @ A_obs_pre) / (n_members - 1.0)
    
    # Obs covariance: Shape (10, 10)
    C_oo_pre = (A_obs_pre.T @ A_obs_pre) / (n_members - 1.0)
    
    # 3. Innovation-based adaptive inflation
    # Expected innovation variance = C_oo + R
    # If observed innovation variance is smaller than expected, ensemble is overconfident -> inflate
    innovation_pre = obs - ens_mean[obs_idx]
    innov_var_obs = np.var(innovation_pre, ddof=1)  # scalar
    innov_var_exp = np.mean(np.diag(C_oo_pre)) + R_val
    
    # Ratio: if observed innov var < expected, we need to inflate
    # inflation^2 should make expected match observed
    # (inflation^2 * C_oo + R) should relate to observed
    # Simple: inflation = sqrt(max(1, innov_var_exp / max(innov_var_obs, eps)))
    # But we want a gentle, bounded adjustment
    eps = 1e-10
    ratio_innov = innov_var_exp / np.maximum(innov_var_obs, eps)
    # Only inflate when ratio > 1 (under-dispersed), deflate gently when ratio < 0.8
    if ratio_innov > 1.0:
        inflation_raw = np.sqrt(ratio_innov)
    else:
        # Mild deflation if over-dispersed, but not below 0.95
        inflation_raw = np.sqrt(max(ratio_innov, 0.7))
    
    # Blend with a small base inflation for numerical safety
    inflation = 0.7 * inflation_raw + 0.3 * 1.01
    inflation = np.clip(inflation, 0.95, 1.15)
    
    # Apply inflation
    ens = ens_mean + inflation * ens_centered
    
    # 4. Recompute anomalies after inflation
    A = ens - ens_mean  # Shape (16, 20)
    A_obs = A[:, obs_idx]  # Shape (16, 10)
    
    # 5. Compute Covariances (post-inflation)
    C_xo = (A.T @ A_obs) / (n_members - 1.0)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    
    # 6. Localization - slightly wider for better information propagation
    sigma = 3.5
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (20, 10)
    
    # Apply localization to cross-covariance
    C_xo_loc = C_xo * loc_weights
    
    # 7. Compute Gain with regularization
    reg = 1e-6
    S = C_oo + R_val * np.eye(n_obs) + reg * np.eye(n_obs)
    
    try:
        K = np.linalg.solve(S, C_xo_loc.T).T
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
        
    # 8. Update Mean
    innovation = obs - ens_mean[obs_idx]  # Shape (10,)
    delta_mean = K @ innovation
    ens_mean = ens_mean + delta_mean
    
    # 9. Update Anomalies
    A_obs_Kt = A_obs @ K.T  # Shape (16, 20)
    A_new = A - A_obs_Kt
    
    # 10. Reconstruct Ensemble
    ens = ens_mean[None, :] + A_new
    
    # 11. Clip and Check
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
