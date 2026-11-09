"""Lorenz-96 ensemble assimilation with stochastic EnKF, inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses observation perturbation, adaptive inflation, and Gaussian localization.
    
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
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # --- Adaptive Multiplicative Inflation (lighter, to complement stochastic update) ---
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target spread: slightly larger than observation std
    target_spread = np.sqrt(R_val) * 1.3
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    
    # Inflation factor based on spread ratio
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation = np.clip(np.mean(ratio), 0.98, 1.12)
    # Blend with a very small base inflation for stability
    inflation = 0.7 * 1.01 + 0.3 * inflation
    inflation = np.clip(inflation, 1.0, 1.10)
    
    # Apply inflation
    ens = ens_mean + inflation * ens_centered
    
    # Recompute mean after inflation
    ens_mean = np.mean(ens, axis=0)
    
    # --- Localization weights ---
    # Gaussian localization with sigma tuned for Lorenz-96
    sigma = 3.0
    indices = np.arange(n_vars)
    # Circular distance
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Localization weights: (n_vars, n_obs)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Stochastic EnKF with Observation Perturbation ---
    # Generate random number generator with a fixed seed for reproducibility
    # But since we don't have access to a global state, use a deterministic approach
    # We'll use the ensemble mean as a seed source for reproducibility
    
    # Create a deterministic pseudo-random sequence based on ensemble statistics
    # This ensures reproducibility across calls
    seed_val = int(np.sum(np.abs(ens_mean)) * 1000000) % (2**31)
    rng = np.random.RandomState(seed_val)
    
    # Generate observation perturbations: (n_members, n_obs)
    # Each member gets a different perturbation
    obs_pert = rng.normal(0.0, np.sqrt(R_val), size=(n_members, n_obs))
    
    # Perturbed observations: (n_members, n_obs)
    obs_perturbed = obs[None, :] + obs_pert
    
    # Compute anomalies
    A_full = ens - ens_mean  # (n_members, n_vars)
    A_obs = ens[:, obs_idx] - ens_mean[obs_idx]  # (n_members, n_obs)
    
    # Compute innovation for each member: (n_members, n_obs)
    innovation = obs_perturbed - ens[:, obs_idx]
    
    # Compute observation-space covariance: S = P_oo + R
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Cholesky decomposition of S
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        S_reg = S + jitter * np.eye(n_obs)
        L = np.linalg.cholesky(S_reg)
    
    # Cross-covariance P_xo: (n_vars, n_obs)
    P_xo = (A_full.T @ A_obs) / (n_members - 1.0)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # Gain: K = P_xo_loc @ S^{-1}, shape (n_vars, n_obs)
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv
    
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # Update each member: ens_new = ens + K @ innovation.T (for each member)
    # innovation shape: (n_members, n_obs)
    # K shape: (n_vars, n_obs)
    # K @ innovation.T gives (n_vars, n_members), transpose to (n_members, n_vars)
    updates = (K @ innovation.T).T  # (n_members, n_vars)
    
    # New ensemble
    ens = ens + updates
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
