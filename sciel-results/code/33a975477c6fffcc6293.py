"""Lorenz-96 ensemble assimilation with LESRF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (LESRF) analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation and Gaussian localization.
    
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
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # --- Adaptive Multiplicative Inflation ---
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target spread: balance between observation noise and model uncertainty
    target_spread = np.sqrt(R_val) * 1.35
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-10))
    
    ratio = target_spread / np.maximum(current_spread, 1e-10)
    mean_ratio = np.mean(ratio)
    
    # Smooth tanh-based inflation to avoid abrupt changes
    # Maps ratio 1.0 -> 1.0, ratio > 1 -> slightly above 1, ratio < 1 -> slightly below
    raw_inflation = 1.0 + 0.15 * np.tanh(3.0 * (mean_ratio - 1.0))
    
    # Blend with a small base inflation for stability
    base_inflation = 1.01
    inflation = 0.6 * base_inflation + 0.4 * raw_inflation
    inflation = np.clip(inflation, 0.99, 1.12)
    
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    
    # --- Localization ---
    # Gaussian localization with sigma tuned for Lorenz-96 correlation length
    sigma = 2.5
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (20, n_obs)
    # Ensure observed locations get full weight
    for i, oi in enumerate(obs_idx):
        loc_weights[oi, i] = 1.0
    
    # --- LESRF Update ---
    # Step 1: Shift observations to zero mean (remove ensemble mean from obs)
    obs_shifted = obs - ens_mean[obs_idx]
    
    # Step 2: Normalize anomalies at observed locations
    A_obs = A[:, obs_idx]  # Shape (n_members, n_obs)
    
    # Compute the square root of the observation covariance
    # S = (1/(M-1)) * A_obs^T @ A_obs + R
    # We use a Cholesky-based approach for numerical stability
    
    # Build the innovation matrix
    # Y = (1/sqrt(M-1)) * A_obs + R^{1/2} * (random or structured)
    # For LESRF, we use the following approach:
    
    # Compute cross-covariance with localization
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # Shape (n_vars, n_obs)
    C_xo_loc = C_xo * loc_weights
    
    # Observation covariance
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # Shape (n_obs, n_obs)
    S = C_oo + R_val * np.eye(n_obs)
    
    # Compute gain using Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        # K = C_xo_loc @ inv(S) = C_xo_loc @ inv(L^T) @ inv(L)
        # Solve L^T X = C_xo_loc^T, then L Y = X, K = Y^T
        X = np.linalg.solve(L.T, C_xo_loc.T)
        Y = np.linalg.solve(L, X)
        K = Y.T  # Shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pinv
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # Update mean
    delta_mean = K @ obs_shifted
    ens_mean = ens_mean + delta_mean
    
    # Update anomalies using square-root formulation
    # A_new = A - A_obs @ K^T
    A_obs_Kt = A_obs @ K.T  # Shape (n_members, n_vars)
    A_new = A - A_obs_Kt
    
    # Reconstruct ensemble
    ens = ens_mean[None, :] + A_new
    
    # --- Spread preservation ---
    # Prevent excessive deflation by restoring a minimum spread at observed locations
    post_mean = np.mean(ens, axis=0)
    post_centered = ens - post_mean
    post_var_obs = np.var(ens[:, obs_idx], axis=0)
    pre_var_obs = np.var(A[:, obs_idx], axis=0) + np.var(ens_mean[obs_idx] - ens_mean[obs_idx], axis=0)  # use pre-update anomaly variance
    # Use the inflation-adjusted pre-update variance as reference
    pre_var_ref = np.var((inflation * (ensemble - np.mean(ensemble, axis=0)))[:, obs_idx], axis=0)
    
    min_spread_factor = 0.95
    post_spread = np.sqrt(np.maximum(post_var_obs, 1e-10))
    ref_spread = np.sqrt(np.maximum(pre_var_ref, 1e-10))
    
    # Only correct if spread dropped too much
    for j, oi in enumerate(obs_idx):
        if post_spread[j] < min_spread_factor * ref_spread[j]:
            scale = min_spread_factor * ref_spread[j] / max(post_spread[j], 1e-10)
            ens[:, oi] = post_mean[oi] + scale * post_centered[:, oi]
    
    # Final clip and check for finiteness
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
