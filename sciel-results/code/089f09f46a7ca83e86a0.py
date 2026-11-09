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
    
    # 1. Adaptive Multiplicative Inflation
    # Compute inflation based on ratio of ensemble spread to expected spread
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # Estimate ensemble variance at observed locations
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)  # shape (10,)
    
    # Expected variance from observations: ensemble spread should be comparable to R
    # If ensemble is too spread out, deflate; if too tight, inflate
    # Target: ensemble spread should be somewhat larger than sqrt(R)
    target_spread = np.sqrt(R_val) * 1.5
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-10))
    
    # Ratio of target to current spread, clamped
    ratio = target_spread / np.maximum(current_spread, 1e-10)
    # Use a smoothed, bounded inflation factor
    inflation = np.clip(np.mean(ratio), 0.9, 1.15)
    # Blend with a base inflation for stability
    inflation = 0.5 * 1.03 + 0.5 * inflation
    inflation = np.clip(inflation, 1.0, 1.12)
    
    ens = ens_mean + inflation * ens_centered
    
    # 2. Compute Anomalies
    A = ens - ens_mean  # Shape (16, 20)
    
    # 3. Compute Covariances
    A_obs = A[:, obs_idx]  # Shape (16, 10)
    
    # Cross-covariance: Shape (20, 10)
    C_xo = (A.T @ A_obs) / (n_members - 1.0)
    
    # Obs covariance: Shape (10, 10)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    
    # 4. Localization
    # Gaussian localization with tighter sigma for Lorenz-96
    sigma = 2.8
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (20, 10)
    
    # Apply localization to cross-covariance
    C_xo_loc = C_xo * loc_weights
    
    # 5. Compute Gain with regularization
    # Add small diagonal regularization for numerical stability
    reg = 1e-6
    S = C_oo + R_val * np.eye(n_obs) + reg * np.eye(n_obs)
    
    try:
        K = np.linalg.solve(S, C_xo_loc.T).T
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
        
    # 6. Update Mean
    innovation = obs - ens_mean[obs_idx]  # Shape (10,)
    delta_mean = K @ innovation
    ens_mean = ens_mean + delta_mean
    
    # 7. Update Anomalies
    A_obs_Kt = A_obs @ K.T  # Shape (16, 20)
    A_new = A - A_obs_Kt
    
    # 8. Reconstruct Ensemble
    ens = ens_mean[None, :] + A_new
    
    # 9. Clip and Check
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
