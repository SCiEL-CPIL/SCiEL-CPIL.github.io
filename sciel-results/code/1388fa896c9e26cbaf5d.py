"""Lorenz-96 ensemble assimilation with sequential LESRF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Sequential Local Ensemble Square Root Filter (LESRF) analysis update for Lorenz-96.
    Uses per-observation sequential updates, tanh-based adaptive inflation, and Gaussian localization.
    
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
    deadline = start_time + 108.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Adaptive Multiplicative Inflation (tanh-based, gentler) ---
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target: maintain spread at ~95% of expected (sqrt(R) * factor)
    target_spread = np.sqrt(R_val) * 1.3
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    
    # Use tanh for smooth, bounded inflation adjustment
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    # tanh maps ratio-1 to a smooth bounded correction
    inflation_correction = np.tanh(0.5 * (np.mean(ratio) - 1.0))
    inflation = 1.0 + 0.3 * inflation_correction
    # Clip to safe range
    inflation = np.clip(inflation, 1.0, 1.12)
    
    # Apply inflation
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    
    # --- Sequential LESRF: update one observation at a time ---
    # This avoids the rank-deficiency issues of simultaneous update with 16 members
    sigma = 2.5  # Localization length scale tuned for L96
    indices = np.arange(n_vars)
    
    for k in range(n_obs):
        if time.time() > deadline:
            break
            
        obs_loc = obs_idx[k]
        obs_val = obs[k]
        
        # Circular distance from this observation location
        dist = np.abs(indices - obs_loc)
        dist = np.minimum(dist, n_vars - dist)
        loc_w = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars,)
        
        # Ensemble anomalies at all locations
        A = ens - ens_mean  # (n_members, n_vars)
        
        # Observation anomaly vector (n_members,)
        A_obs = A[:, obs_loc]
        
        # Innovation
        innov = obs_val - ens_mean[obs_loc]
        
        # Local variance at observation point
        P_oo = np.var(A_obs)
        S = P_oo + R_val
        
        if S < 1e-12:
            continue
            
        # Gain for each variable (with localization)
        P_xo = np.cov(A, A_obs, bias=False, rowvar=False) if n_members > 2 else np.zeros(n_vars)
        # Actually compute properly
        P_xo = (A.T @ A_obs) / (n_members - 1.0)  # (n_vars,)
        P_xo_loc = P_xo * loc_w
        
        K = P_xo_loc / S  # (n_vars,)
        
        # Update mean
        delta_mean = K * innov
        ens_mean = ens_mean + delta_mean
        
        # Update anomalies: A_new = A - A_obs * K
        A = A - np.outer(A_obs, K)
        
        # Reconstruct
        ens = ens_mean[None, :] + A
    
    # --- Final spread check and gentle correction ---
    ens_mean_final = np.mean(ens, axis=0)
    ens_centered_final = ens - ens_mean_final
    final_var_obs = np.var(ens[:, obs_idx], axis=0)
    final_spread = np.sqrt(np.maximum(final_var_obs, 1e-12))
    target_final = np.sqrt(R_val) * 1.1
    spread_ratio = target_final / np.maximum(final_spread, 1e-12)
    
    # Only apply if spread is too low (deflation)
    if np.mean(spread_ratio) > 1.05:
        correction = np.clip(1.0 + 0.2 * np.tanh(0.3 * (np.mean(spread_ratio) - 1.0)), 1.0, 1.08)
        ens = ens_mean_final + correction * ens_centered_final
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
