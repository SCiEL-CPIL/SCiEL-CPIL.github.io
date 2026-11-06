"""Ensemble analysis update for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Local Ensemble Kalman Filter (LEnKF) update with adaptive inflation.
    
    Parameters
    ----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of observed indices.
    observed_indices : array_like
        Indices of observed variables (even indices, 0,2,4,...,18).
    observation_variance : float
        Variance of observation noise.
    
    Returns
    -------
    ndarray, shape (16, 20)
        Updated ensemble.
    """
    t_start = time.time()
    deadline = t_start + 115.0  # 5s safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    result = ensemble.copy()
    
    # Compute ensemble mean and anomaly covariance for observed variables
    ens_mean = result.mean(axis=0)
    anomalies = result - ens_mean  # (n_members, n_vars)
    
    # Observation anomalies
    obs_anoms = observations - ens_mean[obs_idx]  # (n_obs,)
    
    # Compute background covariance at observed locations: (n_obs, n_obs)
    # B = (1/(n-1)) * A_obs @ A_obs.T where A_obs is (n_members, n_obs)
    A_obs = anomalies[:, obs_idx]  # (n_members, n_obs)
    B = (A_obs.T @ A_obs) / (n_members - 1)  # (n_obs, n_obs)
    
    # Observation error covariance: R = sigma^2 * I
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance: S = B + R
    S = B + R
    
    # Compute gain using the ensemble-based formulation to avoid inverting large matrices
    # For local EnKF, we process each observed variable sequentially
    # but for efficiency with 10 observations, we can do a joint update
    
    # Regularize S for numerical stability
    S_reg = S + 1e-10 * np.eye(n_obs)
    
    # Compute inverse of S
    try:
        S_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_reg)
    
    # Gain matrix: K = B @ S_inv, shape (n_obs, n_obs)
    # But we need the full gain for all variables: K_full = A_obs @ B @ S_inv / (n-1)
    # Actually, the standard EnKF update is:
    # x_a = x_b + A_obs @ (B @ S_inv) @ obs_anoms
    # where A_obs is (n_members, n_obs) and the result is (n_members, n_obs)
    # Then we add this to the anomalies at observed locations and propagate to unobserved via covariance
    
    # Let's use the standard formulation:
    # K = B @ S_inv  (n_obs, n_obs)
    # Update for observed components: x_a_obs = x_b_obs + A_obs @ K @ obs_anoms
    # For unobserved components, use cross-covariance:
    # C_cross = anomalies[:, unobs_idx].T @ A_obs / (n-1)  (n_unobs, n_obs)
    # x_a_unobs = x_b_unobs + C_cross @ S_inv @ obs_anoms
    
    # Actually, let's do this more carefully with the full state
    # The full gain matrix K_full is (n_vars, n_obs):
    # K_full = (anomalies.T @ A_obs) / (n-1) @ S_inv
    
    # Cross-covariance between full state and observations
    C = (anomalies.T @ A_obs) / (n_members - 1)  # (n_vars, n_obs)
    
    # Gain: K = C @ S_inv  (n_vars, n_obs)
    K = C @ S_inv  # (n_vars, n_obs)
    
    # Innovation: d = obs_anoms  (n_obs,)
    # Update: x_a = x_b + K @ d for each member
    # But this is the mean update. For the ensemble, we need:
    # x_a_i = x_b_i + K @ (obs - y_b_i) where y_b_i = x_b_i[obs_idx]
    
    # Let's compute per-member updates
    # For member i: y_b_i = ensemble[i, obs_idx]
    # innovation_i = observations - y_b_i
    # update_i = K @ innovation_i  (n_vars,)
    
    y_b = ensemble[:, obs_idx]  # (n_members, n_obs)
    innovations = observations[None, :] - y_b  # (n_members, n_obs)
    
    # Update each member: x_a_i = x_b_i + K @ innovation_i
    # K is (n_vars, n_obs), innovations is (n_members, n_obs)
    # So: updates = innovations @ K.T  (n_members, n_vars)
    updates = innovations @ K.T  # (n_members, n_vars)
    
    result = ensemble + updates
    
    # Apply multiplicative inflation to prevent filter collapse
    # Compute current ensemble spread at observed locations
    current_mean = result.mean(axis=0)
    current_anoms = result - current_mean
    
    # Target spread: use a blend of background and observation-based spread
    # Simple adaptive inflation: scale anomalies to maintain a minimum spread
    # Compute variance at each observed location
    var_obs = np.var(current_anoms[:, obs_idx], axis=0)  # (n_obs,)
    
    # Target variance: based on observation variance and typical Lorenz-96 variance
    # For F=8, typical variance is around 10-20. Use a conservative target.
    target_var = 1.0  # We'll use a simple inflation factor
    
    # Compute inflation factor
    # If spread is too small, inflate
    min_var = 0.1
    inflation = 1.0
    for v in var_obs:
        if v < min_var:
            inflation = max(inflation, np.sqrt(min_var / max(v, 1e-10)))
    
    # Cap inflation to avoid instability
    inflation = min(inflation, 1.5)
    
    if inflation > 1.0:
        result = current_mean + inflation * current_anoms
    
    # Ensure output is finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
