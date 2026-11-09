"""Lorenz-96 ensemble assimilation with Stochastic Ensemble Kalman Filter."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses random rotation matrices for unbiased stochastic perturbations,
    with adaptive inflation and Gaussian localization.
    
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
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean  # (n_members, n_vars)
    
    # --- Adaptive Multiplicative Inflation ---
    # Inflate to maintain adequate spread
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    target_spread = np.sqrt(R_val) * 1.4
    
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation_factor = np.median(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.12)
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_anom
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean
    
    # --- Localization weights ---
    sigma = 2.5  # Localization scale for Lorenz-96 correlation structure
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    
    # --- Stochastic Ensemble Kalman Filter Update ---
    # Use random rotation for unbiased stochastic perturbations
    
    # Anomalies at observed locations
    A_obs = ens_anom[:, obs_idx]  # (n_members, n_obs)
    
    # Innovation
    innovation = obs - ens_mean[obs_idx]  # (n_obs,)
    
    # Cross-covariance between all variables and observations
    P_xo = (ens_anom.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization
    P_xo_loc = P_xo * loc_weights
    
    # Observation error covariance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Cholesky of S
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        S_reg = S + jitter * np.eye(n_obs)
        L = np.linalg.cholesky(S_reg)
    
    # Whitened innovation
    y = np.linalg.solve(L, innovation)  # (n_obs,)
    
    # Compute whitened anomalies: W = A_obs @ L^{-T}
    L_inv = np.linalg.inv(L)
    W = A_obs @ L_inv.T  # (n_members, n_obs)
    
    # Generate random rotation matrix (Hadamard-based or QR of random normal)
    # Use QR decomposition of random matrix for rotation
    rng = np.random.default_rng(42)
    Q = np.linalg.qr(rng.standard_normal((n_members, n_members)))[0]
    
    # Stochastic perturbation: for each member, add rotated whitened anomaly
    # x_a_i = x_f_i + K @ (y - W_i)  where W_i is row i of W
    # This is equivalent to: x_a_i = x_f_i + K @ y - K @ W_i
    
    # Gain in whitened space: K_w = P_xo_loc @ L^{-T}  (n_vars, n_obs)
    K_w = P_xo_loc @ L_inv.T  # (n_vars, n_obs)
    
    # Update: x_a = x_f + K_w @ (y - Q^T @ W^T)^T
    # = x_f + K_w @ y - K_w @ (Q @ W^T)^T
    # = x_f + K_w @ y - (K_w @ Q @ W^T)^T
    
    # Compute update term
    update_mean = K_w @ y  # (n_vars,) - the deterministic part
    
    # Stochastic part: for each member i, subtract K_w @ (Q @ W^T)[i, :]
    # (Q @ W^T) is (n_members, n_obs), so (K_w @ Q @ W^T) is (n_vars, n_members)
    stochastic_part = K_w @ (Q @ W.T)  # (n_vars, n_members)
    
    # Apply update to each member
    ens = ens + update_mean[None, :] - stochastic_part.T  # (n_members, n_vars)
    
    # --- Post-processing: ensure finite and bounded ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
