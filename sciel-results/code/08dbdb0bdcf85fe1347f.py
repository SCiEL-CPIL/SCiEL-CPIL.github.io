"""Lorenz-96 ensemble assimilation: Stochastic EnKF with adaptive inflation and tight localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter with adaptive inflation and tight localization.
    
    Uses observation perturbation to prevent variance collapse,
    adaptive multiplicative inflation based on innovation statistics,
    and Gaussian localization for the cross-covariance.
    
    Parameters
    ----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns
    -------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # Shape (16, 20)
    
    # 2. Adaptive inflation based on innovation statistics
    A_obs = A[:, obs_idx]  # Shape (16, n_obs)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    S = C_oo + R_val * np.eye(n_obs)
    
    innovation = obs - ens_mean[obs_idx]
    expected_innov_var = np.trace(S) / n_obs
    actual_innov_var = np.dot(innovation, innovation) / n_obs
    
    # Adaptive inflation: gentle, innovation-based
    ratio = actual_innov_var / (expected_innov_var + 1e-10)
    # Smooth adaptive inflation: small adjustments
    base_inflation = 1.01
    if ratio > 1.0:
        inflation = base_inflation + 0.015 * min(ratio - 1.0, 3.0)
    else:
        inflation = base_inflation - 0.005 * min(1.0 - ratio, 1.0)
    inflation = np.clip(inflation, 1.0, 1.08)
    
    # Apply multiplicative inflation to anomalies
    A = inflation * A
    ens = ens_mean + A
    
    # 3. Stochastic observation perturbation
    # Generate random perturbations for each member's observations
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within call
    # Perturb observations: obs_pert[i] = obs + sqrt(R) * N(0, I)
    sqrt_R = np.sqrt(R_val)
    obs_pert = obs[None, :] + sqrt_R * rng.standard_normal((n_members, n_obs))
    
    # 4. Tight Gaussian localization
    sigma_loc = 2.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma_loc) ** 2)  # Shape (20, n_obs)
    
    # 5. Compute localized cross-covariance
    C_xo = (A.T @ A[:, obs_idx]) / (n_members - 1.0)  # Shape (20, n_obs)
    C_xo_loc = C_xo * loc_weights
    
    # 6. Compute Kalman gain
    try:
        K = np.linalg.solve(S, C_xo_loc.T).T  # Shape (20, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # 7. Update each member using perturbed observations
    ens_new = np.empty_like(ens)
    for i in range(n_members):
        # Innovation for this member
        innov_i = obs_pert[i] - (ens_mean + A[i])[obs_idx]
        # Update: x_i_new = x_i + K @ innov_i
        x_i = ens_mean + A[i]
        ens_new[i] = x_i + K @ innov_i
    
    # 8. Clip and ensure finite
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.where(np.isfinite(ens_new), ens_new, 0.0)
    
    if time.time() > deadline:
        return ens_new.astype(ensemble.dtype)
    
    return ens_new.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
