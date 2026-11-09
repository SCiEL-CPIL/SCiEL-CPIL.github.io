"""Lorenz-96 ensemble assimilation with Stochastic EnKF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation, Gaussian localization, and
    observation perturbation (stochastic) update.
    
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
    deadline = start_time + 115.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Step 1: Compute ensemble statistics ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean  # (n_members, n_vars)
    
    # --- Step 2: Adaptive Multiplicative Inflation ---
    # Innovation-based inflation
    innovation = obs - ens_mean[obs_idx]  # (n_obs,)
    P_yy = np.mean(ens_anom[:, obs_idx] ** 2, axis=0)  # (n_obs,)
    expected_innov_var = np.mean(P_yy + R_val)
    innovation_var = np.mean(innovation ** 2)
    
    if expected_innov_var > 1e-10:
        ratio_innov = innovation_var / expected_innov_var
        inflation_adaptive = np.clip(ratio_innov, 0.95, 1.25)
    else:
        inflation_adaptive = 1.02
    
    # Spread-based inflation
    current_spread = np.sqrt(np.maximum(np.var(ens[:, obs_idx], axis=0), 1e-12))
    target_spread = np.sqrt(R_val) * 1.3
    spread_ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation_spread = np.clip(np.mean(spread_ratio), 0.98, 1.12)
    
    # Blend: weight innovation-based more heavily
    inflation_factor = 0.6 * inflation_adaptive + 0.4 * inflation_spread
    # Add small base inflation for stability
    inflation_factor = 0.3 * 1.01 + 0.7 * inflation_factor
    inflation_factor = np.clip(inflation_factor, 1.0, 1.10)
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_anom
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean
    
    # --- Step 3: Localization Weights ---
    indices = np.arange(n_vars)
    # Circular distance for periodic boundary
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Gaussian localization with sharper scale
    sigma = 2.2
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    
    # --- Step 4: Stochastic EnKF Update with Observation Perturbation ---
    # Anomalies at observed locations
    A_obs = ens_anom[:, obs_idx]  # (n_members, n_obs)
    
    # Cross-covariance between all variables and observations
    P_xo = (ens_anom.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # Observation-space covariance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Ensure positive definiteness
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        jitter = 1e-6
        L = np.linalg.cholesky(S + jitter * np.eye(n_obs))
    
    # S^{-1} via Cholesky
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv
    
    # Kalman gain
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # --- Step 5: Observation Perturbation ---
    # Use a deterministic seed based on ensemble hash for reproducibility
    seed = hash((tuple(ens.ravel()), tuple(obs.ravel()))) % (2**32)
    rng = np.random.default_rng(seed)
    
    R_std = np.sqrt(R_val)
    perturbed_obs = obs[None, :] + R_std * rng.standard_normal((n_members, n_obs))
    
    # --- Step 6: Update each member ---
    ens_obs = ens[:, obs_idx]  # (n_members, n_obs)
    innovation_per_member = perturbed_obs - ens_obs  # (n_members, n_obs)
    
    # Update each member
    updates = innovation_per_member @ K.T  # (n_members, n_vars)
    ens_updated = ens + updates
    
    # --- Step 7: Post-processing ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.where(np.isfinite(ens_updated), ens_updated, 0.0)
    
    if time.time() > deadline:
        return ens_updated.astype(ensemble.dtype)
    
    return ens_updated.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
