"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with:
    - Multiplicative inflation to prevent filter collapse
    - Exponential localization of cross-covariance
    - Scaled analysis step
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors
    observations : ndarray, shape (16, 10)
        Noisy observations of even-indexed variables
    observed_indices : array-like, shape (10,)
        Indices of observed variables (even indices: 0,2,4,...,18)
    observation_variance : float
        Variance of observation noise
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis
    """
    start_time = time.time()
    
    # Deadline guard: ensure we don't exceed a safe fraction of the 150s budget
    # Each call should be fast; this is a safety net
    deadline = start_time + 10.0  # 10 seconds max per call
    
    M, N = ensemble.shape  # M=16 members, N=20 variables
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Convert to float64 for numerical stability
    ens = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64)
    R = float(observation_variance)
    
    # --- Step 1: Multiplicative Inflation ---
    # Compute ensemble mean and spread
    ens_mean = ens.mean(axis=0)
    
    # Adaptive inflation: adjust inflation factor based on ensemble spread vs observation error
    # Compute innovation (observation - ensemble mean at observed locations)
    innovation = obs - ens[:, obs_idx]
    
    # Compute ensemble variance at observed locations
    ens_var_obs = ((ens[:, obs_idx] - ens_mean[obs_idx])**2).mean(axis=0)
    
    # Compute observation variance (scalar or vector)
    if np.isscalar(R) or np.ndim(R) == 0:
        R_vec = np.full(n_obs, float(R))
    else:
        R_vec = np.asarray(R, dtype=np.float64)
        if R_vec.ndim == 0:
            R_vec = np.full(n_obs, float(R_vec))
        elif R_vec.shape[0] != n_obs:
            R_vec = np.full(n_obs, float(R))
    
    # Adaptive inflation factor
    # If ensemble spread is too small relative to observation error, inflate
    # Use a simple adaptive scheme
    spread_ratio = ens_var_obs / R_vec
    # Target spread ratio ~1; if spread is too small, inflate more
    # inflation_factor = 1 + 0.1 * np.maximum(0, 1 - spread_ratio).mean()
    
    # Use a fixed mild inflation plus adaptive component
    base_inflation = 1.01
    adaptive_factor = 1.0 + 0.05 * np.clip(1.0 - spread_ratio.mean(), 0, 2)
    inflation = base_inflation * adaptive_factor
    
    # Apply multiplicative inflation about the mean
    ens = ens_mean + inflation * (ens - ens_mean)
    
    # --- Step 2: Compute Ensemble Covariance and Cross-Covariance ---
    # Centered ensemble
    ens_centered = ens - ens.mean(axis=0)
    
    # Ensemble covariance (state-state)
    P = (ens_centered.T @ ens_centered) / (M - 1)
    
    # Cross-covariance (state-observed)
    # P_xo[i,j] = cov(x_i, y_j)
    obs_centered = obs - ens[:, obs_idx].mean(axis=0)
    P_xo = (ens_centered.T @ obs_centered) / (M - 1)
    
    # --- Step 3: Localization ---
    # Exponential localization: reduce cross-covariance with distance
    # For Lorenz-96, variables are on a ring with periodic boundary
    # Distance between variable i and observed variable j
    i_idx = np.arange(N)
    j_idx = obs_idx
    
    # Compute periodic distance
    dist = np.abs(i_idx[:, None] - j_idx[None, :])
    dist = np.minimum(dist, N - dist)  # periodic distance
    
    # Localization length scale (tune this)
    L = 4.0  # localization length in grid units
    loc_weights = np.exp(-0.5 * (dist / L)**2)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # --- Step 4: Compute Kalman Gain ---
    # Observation error covariance: P_yy + R
    # P_yy = cov(observations, observations) = (ens[:, obs_idx] - mean)^T (ens[:, obs_idx] - mean) / (M-1)
    ens_obs = ens[:, obs_idx]
    ens_obs_mean = ens_obs.mean(axis=0)
    P_yy = ((ens_obs - ens_obs_mean).T @ (ens_obs - ens_obs_mean)) / (M - 1)
    
    S = P_yy + np.diag(R_vec)
    
    # Kalman gain: K = P_xo S^{-1}
    # Use solve for numerical stability
    try:
        S_inv = np.linalg.inv(S)
        K = P_xo_loc @ S_inv
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv
    
    # --- Step 5: Apply Analysis Update ---
    # For each ensemble member, update:
    # x_a = x_f + K (y - H x_f)
    # where H x_f = x_f[obs_idx]
    
    # Innovation for each member
    innovations = obs[None, :] - ens[:, obs_idx]  # shape (M, n_obs)
    
    # Analysis update
    ens_updated = ens + innovations @ K.T
    
    # --- Step 6: Additive Inflation (optional, to maintain spread) ---
    # Small additive inflation to counteract any remaining collapse
    ens_updated_mean = ens_updated.mean(axis=0)
    ens_updated = ens_updated_mean + 1.005 * (ens_updated - ens_updated_mean)
    
    # --- Step 7: Clip to valid range ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    
    # Ensure finiteness
    ens_updated = np.nan_to_num(ens_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() > deadline:
        # Return what we have
        pass
    
    return ens_updated
# EVOLVE-BLOCK-END
