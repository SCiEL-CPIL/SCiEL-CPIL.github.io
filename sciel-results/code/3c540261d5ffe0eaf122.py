"""Ensemble Kalman Filter with localization and adaptive inflation for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Kalman Filter (EnKF) analysis step.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like, shape (10,)
        Indices of the observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite inputs
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0)
    observations = np.where(np.isfinite(observations), observations, 0.0)
    
    # Compute ensemble mean and covariance
    mean = np.mean(ensemble, axis=0)
    
    # Anomalies: centered ensemble
    anomalies = ensemble - mean
    
    # Observation operator: select observed components
    obs_mean = mean[obs_idx]
    obs_anoms = anomalies[:, obs_idx]  # shape (n_members, n_obs)
    
    # Innovation: observations - ensemble mean at observed locations
    innovation = observations - obs_mean  # shape (n_obs,)
    
    # Observation error covariance (diagonal)
    R = np.eye(n_obs) * observation_variance
    
    # Background covariance at observed locations
    # S_obs = H P H^T + R
    # H P H^T = (1/(n-1)) * obs_anoms @ obs_anoms.T
    S_obs = (obs_anoms @ obs_anoms.T) / (n_members - 1) + R
    
    # Compute Kalman gain K = P H^T S_obs^{-1}
    # P H^T = (1/(n-1)) * anomalies @ obs_anoms.T  (shape n_vars x n_obs)
    PHt = (anomalies @ obs_anoms.T) / (n_members - 1)
    
    # Solve S_obs @ K^T = PHt  =>  K^T = S_obs^{-1} @ PHt
    try:
        Kt = np.linalg.solve(S_obs, PHt)  # shape (n_obs, n_vars)
        K = Kt.T  # shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        Kt = np.linalg.pinv(S_obs) @ PHt
        K = Kt.T
    
    # Localization: reduce weights of distant observations
    # Use exponential localization in index space
    loc_range = 5.0  # decay scale
    for i, oi in enumerate(obs_idx):
        for j in range(n_vars):
            dist = min(abs(j - oi), n_vars - abs(j - oi))
            K[j, i] *= np.exp(-0.5 * (dist / loc_range) ** 2)
    
    # Update ensemble: x_new = x + K * (y - H x)
    # For each member: x_i_new = x_i + K @ (y - H x_i)
    # y - H x_i = observations - ensemble[i, obs_idx]
    updates = np.zeros_like(ensemble)
    for m in range(n_members):
        innovation_m = observations - ensemble[m, obs_idx]  # shape (n_obs,)
        updates[m] = K @ innovation_m  # shape (n_vars,)
    
    updated_ensemble = ensemble + updates
    
    # Adaptive inflation: check if spread is too small
    # Compute spread at observed locations
    spread = np.std(updated_ensemble[:, obs_idx], axis=0)
    mean_spread = np.mean(spread)
    
    # If spread is too small relative to observation error, inflate
    target_spread = np.sqrt(observation_variance) * 0.5
    if mean_spread < target_spread:
        inflation_factor = 1.0 + 0.01 * (target_spread - mean_spread) / (target_spread + 1e-10)
        updated_ensemble = mean + (updated_ensemble - mean) * inflation_factor
    
    # Clamp to finite and within bounds
    updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, 0.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
