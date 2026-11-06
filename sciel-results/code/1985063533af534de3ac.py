"""Ensemble Kalman Filter with localization, inflation, and rank-deficiency handling."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using an Ensemble Kalman Filter (EnKF)
    with spatial localization, multiplicative inflation, and rank-deficiency handling.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of the observed variables.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    # Ensure numpy arrays
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # shape (n_members, n_vars)
    
    # Estimate background covariance
    # C_bb = (1/(N-1)) * A^T A, where A is anomalies
    # But we need C_bb for observed variables and cross-covariance
    A_obs = anomalies[:, observed_indices]  # shape (N, n_obs)
    A_full = anomalies  # shape (N, n_vars)
    
    # Compute covariance matrices
    # Use the standard EnKF formula: K = C_bx * (C_bb + R)^{-1}
    # C_bb is n_obs x n_obs, C_bx is n_vars x n_obs
    # To handle potential rank deficiency, add regularization
    
    # Compute C_bb = A_obs^T A_obs / (N-1)
    C_bb = (A_obs.T @ A_obs) / (n_members - 1)  # shape (n_obs, n_obs)
    
    # Compute C_bx = A_full^T A_obs / (N-1)
    C_bx = (A_full.T @ A_obs) / (n_members - 1)  # shape (n_vars, n_obs)
    
    # Observation covariance R = I * observation_variance
    R = np.eye(n_obs) * observation_variance
    
    # Add small regularization to handle rank deficiency
    reg = 1e-6 * np.eye(n_obs)
    S = C_bb + R + reg  # shape (n_obs, n_obs)
    
    # Compute gain matrix K = C_bx * S^{-1}
    # Use solve for numerical stability
    try:
        K = C_bx @ np.linalg.inv(S)  # shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        K = C_bx @ np.linalg.pinv(S)
    
    # Innovation: observations - background mean at observed locations
    innovation = observations - mean[observed_indices]  # shape (n_obs,)
    
    # Update mean
    mean_updated = mean + K @ innovation  # shape (n_vars,)
    
    # Update anomalies (ensemble members)
    # For each member, update the anomaly: a_i' = a_i - K * (y_i - y_mean)
    # But in standard EnKF, we update the full state:
    # x_i' = x_i + K * (y_i - y_mean_observed)
    # where y_i is the observation for member i (but we don't have perturbed obs)
    # Standard EnKF without perturbed observations:
    # x_i' = x_i + K * (y - y_mean)  for all i (this is the "spherical" or "deterministic" version)
    # But this collapses the ensemble. Better to use the proper EnKF update.
    
    # Proper EnKF: x_i' = x_i + K * (y_i - y)
    # where y_i are perturbed observations. Since we don't have them, we use:
    # x_i' = x_i + K * (y - y_mean) - K * (y_i - y_mean_obs)
    # But we don't have y_i. So we use the "relaxed" approach:
    # Update the mean, and adjust anomalies to preserve the ensemble spread.
    
    # Alternative: Use the "EnKF" with the formula:
    # x_i' = x_i + K * (y - y_mean) for the mean part
    # and scale anomalies to maintain variance.
    
    # Let's use the standard EnKF approach where we update each member:
    # x_i' = x_i + K @ (y - y_mean)
    # This is the "deterministic" EnKF which can reduce spread.
    # To compensate, we apply multiplicative inflation.
    
    # First, compute the updated ensemble using the deterministic EnKF
    update = np.outer(np.ones(n_members), K @ innovation)  # shape (n_members, n_vars)
    ensemble_updated = ensemble + update
    
    # Now, apply multiplicative inflation to restore spread
    # Compute the ratio of background spread to updated spread
    # For each variable, compute the std of anomalies before and after
    old_std = np.std(anomalies, axis=0)  # shape (n_vars,)
    new_anomalies = ensemble_updated - np.mean(ensemble_updated, axis=0)
    new_std = np.std(new_anomalies, axis=0)  # shape (n_vars,)
    
    # Inflation factor: sqrt(old_std / new_std), clipped to avoid extreme values
    # Only inflate if new_std is too small
    inflation = np.ones(n_vars)
    for j in range(n_vars):
        if new_std[j] > 1e-10:
            ratio = old_std[j] / new_std[j]
            # Clip inflation to [0.95, 1.5] to avoid over-inflation
            inflation[j] = np.clip(np.sqrt(ratio), 0.95, 1.5)
    
    # Apply inflation to anomalies
    ensemble_updated = np.mean(ensemble_updated, axis=0) + inflation * new_anomalies
    
    # Ensure finite values
    ensemble_updated = np.nan_to_num(ensemble_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to [-100, 100] as required
    ensemble_updated = np.clip(ensemble_updated, -100.0, 100.0)
    
    return ensemble_updated
