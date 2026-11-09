"""Lorenz-96 ensemble assimilation with Stochastic EnKF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-Kalman) analysis update for Lorenz-96.
    Uses random rotation perturbations for unbiased ensemble mean and spread preservation.
    
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
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # Shape (16, 20)
    A_obs = A[:, obs_idx]  # Shape (16, 10)
    
    # 2. Adaptive Multiplicative Inflation (smooth, conservative)
    ens_var_obs = np.var(A_obs, axis=0)  # shape (10,)
    target_spread = np.sqrt(R_val) * 1.4
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-10))
    
    # Use log-sum-exp soft clamp for smooth inflation
    ratio = target_spread / np.maximum(current_spread, 1e-10)
    # Soft clamp: smooth approximation of clip(ratio, 0.9, 1.15)
    # Use a sigmoid-based soft clamp
    lower, upper = 0.9, 1.15
    soft_lower = np.logaddexp(0, np.log(1e-6))  # placeholder, we'll do it differently
    # Simpler: smooth blend
    inflation_raw = np.mean(ratio)
    # Smooth clip via tanh
    inflation = lower + (upper - lower) / (1 + np.exp(-8 * (inflation_raw - 1.0)))
    # Blend with base
    inflation = 0.6 * 1.02 + 0.4 * inflation
    inflation = np.clip(inflation, 1.0, 1.12)
    
    # Apply inflation
    ens = ens_mean + inflation * A
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    A_obs = A[:, obs_idx]
    
    # 3. Localization
    sigma = 2.8
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (20, 10)
    
    # 4. Compute covariances
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # Shape (20, 10)
    C_xo_loc = C_xo * loc_weights
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # Shape (10, 10)
    
    # 5. Innovation and gain
    innovation = obs - ens_mean[obs_idx]  # Shape (10,)
    S = C_oo + R_val * np.eye(n_obs) + 1e-8 * np.eye(n_obs)
    
    try:
        K = np.linalg.solve(S, C_xo_loc.T).T  # Shape (20, 10)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # 6. Stochastic EnKF update
    # Generate random rotation matrix Q (n_obs x n_obs)
    # Use QR decomposition of a random Gaussian matrix
    rng = np.random.default_rng(seed=hash((tuple(obs), tuple(obs_idx), R_val)) % (2**32))
    random_mat = rng.standard_normal((n_obs, n_obs))
    Q, _ = np.linalg.qr(random_mat)
    
    # Perturb observations and ensemble anomalies with Q
    # Perturbed observation: y_pert = y + R^{1/2} @ Q
    R_sqrt = np.sqrt(R_val)
    y_pert = obs + R_sqrt * Q @ np.ones(n_obs)  # This is not quite right
    
    # Correct stochastic EnKF:
    # We need to perturb both observations and the ensemble in observation space
    # such that the ensemble mean of the update equals the deterministic Kalman mean
    
    # Standard stochastic EnKF (Hunt et al. 2007):
    # 1. Generate Q (random rotation)
    # 2. Perturb observations: y_i = y + R^{1/2} Q e_i for each member i
    # 3. For each member, do a local Kalman update with its perturbed observation
    
    # But with only 16 members and 10 obs, we can do:
    # Perturb the observation vector for each member
    # y_pert[m] = obs + R^{1/2} * Q[m, :]  -- but Q is n_obs x n_obs, we need n_members rows
    
    # Actually, the standard approach:
    # Generate a random matrix Q of shape (n_obs, n_members)
    # Then for member m: y_m = obs + R^{1/2} * Q[:, m]
    
    Q_full = rng.standard_normal((n_obs, n_members))
    # Orthogonalize columns (or just use random perturbations)
    # For simplicity, use the QR approach to get orthonormal columns
    Q_orth, _ = np.linalg.qr(Q_full)
    # Q_orth is (n_obs, min(n_obs, n_members)) = (10, 10)
    # We need n_members columns, so if n_members > n_obs, we can't get full orthonormal
    # For n_members=16, n_obs=10, we get 10 orthonormal columns
    # Use the first n_members columns (repeat if needed) or just use random
    
    # Simpler approach: use the deterministic gain but add stochastic perturbation
    # to preserve spread. The key insight: the ensemble mean update is the same,
    # but the anomaly update uses random perturbations.
    
    # Update mean (deterministic part)
    delta_mean = K @ innovation
    ens_mean_new = ens_mean + delta_mean
    
    # For anomalies, use stochastic perturbation
    # A_new[m] = A[m] - K @ (A_obs[m] + perturbation[m])
    # where perturbation is chosen so that mean(A_new) = 0
    
    # Generate perturbations
    # We want: mean_m [K @ (A_obs[m] + pert[m])] = K @ mean_m(A_obs[m]) + K @ mean_m(pert[m])
    # Since mean_m(A_obs[m]) = 0, we need mean_m(pert[m]) = 0
    
    # Use random perturbations with zero mean
    perturb = rng.standard_normal((n_members, n_obs))
    perturb -= np.mean(perturb, axis=0, keepdims=True)  # Zero mean
    # Scale by sqrt(R_val) to match observation noise scale
    perturb *= np.sqrt(R_val)
    
    # Update anomalies
    # A_obs_pert = A_obs + perturb  # Shape (16, 10)
    # A_new = A - K @ A_obs_pert.T  # Shape (20, 16) -> transpose
    A_obs_pert = A_obs + perturb
    A_new = A - (K @ A_obs_pert.T).T  # Shape (16, 20)
    
    # Reconstruct ensemble
    ens = ens_mean_new[None, :] + A_new
    
    # 7. Clip and check
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
