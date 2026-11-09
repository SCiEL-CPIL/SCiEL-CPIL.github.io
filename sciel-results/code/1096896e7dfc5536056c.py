"""Lorenz-96 ensemble assimilation with LESRF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (LESRF) analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation, Gaussian localization, and Cholesky-based
    square-root gain computation for numerical stability.
    
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
    target_spread = np.sqrt(R_val) * 1.4
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    
    # Inflation factor based on spread ratio
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation = np.clip(np.mean(ratio), 0.95, 1.18)
    # Blend with a small base inflation for stability
    inflation = 0.6 * 1.02 + 0.4 * inflation
    inflation = np.clip(inflation, 1.0, 1.15)
    
    # Apply inflation
    ens = ens_mean + inflation * ens_centered
    
    # Recompute mean after inflation
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # (n_members, n_vars)
    
    # --- Localization weights ---
    # Gaussian localization with sigma tuned for Lorenz-96 correlation length
    sigma = 3.0
    indices = np.arange(n_vars)
    # Circular distance
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Localization weights: (n_vars, n_obs)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- LESRF Analysis ---
    # Anomalies at observed locations
    A_obs = A[:, obs_idx]  # (n_members, n_obs)
    
    # Cross-covariance: (n_vars, n_obs)
    C_xo = (A.T @ A_obs) / (n_members - 1.0)
    
    # Apply localization to cross-covariance
    C_xo_loc = C_xo * loc_weights
    
    # Observation covariance: (n_obs, n_obs)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    
    # Innovation covariance: S = C_oo + R
    S = C_oo + R_val * np.eye(n_obs)
    
    # Add small regularization for numerical stability
    S = S + 1e-8 * np.eye(n_obs)
    
    # Cholesky decomposition for stable gain computation
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Add more jitter if not positive definite
        S = S + 1e-6 * np.eye(n_obs)
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            S_inv = np.linalg.pinv(S)
            K = C_xo_loc @ S_inv
            innovation = obs - ens_mean[obs_idx]
            delta_mean = K @ innovation
            ens_mean = ens_mean + delta_mean
            A_obs_Kt = A_obs @ K.T
            A_new = A - A_obs_Kt
            ens = ens_mean[None, :] + A_new
            ens = np.clip(ens, -100.0, 100.0)
            ens = np.where(np.isfinite(ens), ens, 0.0)
            return ens.astype(ensemble.dtype)
    
    # Innovation: (n_obs,)
    innovation = obs - ens_mean[obs_idx]
    
    # Solve L @ y = innovation for y
    y = np.linalg.solve(L, innovation)
    
    # Compute gain: K = C_xo_loc @ S^{-1}
    # S^{-1} = L^{-T} @ L^{-1}
    # We can compute K @ innovation = C_xo_loc @ S^{-1} @ innovation
    # = C_xo_loc @ L^{-T} @ (L^{-1} @ innovation)
    # = C_xo_loc @ L^{-T} @ y
    
    # Solve L^T @ z = y
    z = np.linalg.solve(L.T, y)
    
    # delta_mean = C_xo_loc @ z
    delta_mean = C_xo_loc @ z
    
    # Update mean
    ens_mean = ens_mean + delta_mean
    
    # Update anomalies: A_new = A - A_obs @ K.T
    # K.T has shape (n_obs, n_vars)
    # K.T = S^{-1} @ C_xo_loc.T
    # We need A_obs @ K.T = A_obs @ S^{-1} @ C_xo_loc.T
    # = A_obs @ L^{-T} @ L^{-1} @ C_xo_loc.T
    
    # Solve L @ w = A_obs for w (n_members, n_obs)
    w = np.linalg.solve(L.T, A_obs.T).T  # Actually solve L.T @ w = A_obs
    
    # More carefully: we want A_obs @ L^{-T}
    # Let W = A_obs @ L^{-T}, so L @ W.T = A_obs.T, W.T = L^{-1} @ A_obs.T
    W_T = np.linalg.solve(L, A_obs.T)  # (n_obs, n_members)
    W = W_T.T  # (n_members, n_obs)
    
    # Now A_obs @ S^{-1} = W @ L^{-1}
    # W @ L^{-1} means solve L @ V = W for V
    V_T = np.linalg.solve(L, W.T)  # (n_obs, n_members)
    V = V_T.T  # (n_members, n_obs)
    
    # A_obs @ S^{-1} @ C_xo_loc.T = V @ C_xo_loc.T
    A_obs_Kt = V @ C_xo_loc.T  # (n_members, n_vars)
    
    A_new = A - A_obs_Kt
    
    # Reconstruct ensemble
    ens = ens_mean[None, :] + A_new
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
