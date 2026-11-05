"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Smoother Transform (LEST) for Lorenz-96 assimilation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    N = 16  # ensemble size
    nx = 20  # state dimension
    
    # Ensure proper shapes
    ensemble = np.asarray(ensemble, dtype=float)
    observations = np.asarray(observations, dtype=float)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    # Handle case where observations might be shaped (1, 10) or (10,)
    if observations.ndim == 2:
        observations = observations[0]
    
    # Ensure observations length matches observed_indices length
    nobs = len(observed_indices)
    if len(observations) != nobs:
        # If observations are longer, take only the relevant ones
        observations = observations[:nobs]
    
    # Compute ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - x_mean  # shape (N, nx)
    
    # Compute ensemble covariance at observed locations
    # B_obs = (1/(N-1)) * A_obs @ A_obs.T
    A_obs = anomalies[:, observed_indices]  # shape (N, nobs)
    B_obs = (A_obs @ A_obs.T) / (N - 1)  # shape (nobs, nobs)
    
    # Innovation
    y_innov = observations - x_mean[observed_indices]  # shape (nobs,)
    
    # Observation error covariance
    R = observation_variance * np.eye(nobs)
    
    # Kalman gain computation via square root form
    # S = B_obs + R
    S = B_obs + R
    
    # Compute S^{-1/2} using Cholesky
    try:
        S_chol = np.linalg.cholesky(S)
        S_inv_sqrt = np.linalg.inv(S_chol.T)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1e-10)
        S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Compute transformed innovation
    # K = B_obs @ S^{-1}
    # But we'll use the transform method
    # 
    # The LEST update: x_a = x + A @ S^{-1/2} @ S^{-1/2} @ (y - x_mean_obs)
    # More precisely, we want:
    # x_a_i = x_i + sum_j K_ij (y_j - x_mean_j)
    # where K = B S^{-1}
    #
    # Using square root: K = A @ S^{-1/2} @ S^{-1/2}
    # Actually, the standard EnKF/LEST transform is:
    # Let W = S^{-1/2} @ (y - x_mean_obs)
    # Then x_a = x + A @ S^{-1/2} @ W
    # Wait, let me be more careful.
    #
    # B = A @ A.T / (N-1) where A is (N x nobs)
    # S = B + R
    # K = B @ S^{-1}
    # x_a = x + K @ (y - x_mean_obs)
    #
    # To avoid explicit inverse, use:
    # K = A @ (A.T @ S^{-1} @ A) / (N-1) ... this gets complicated
    #
    # Simpler: just compute K directly for small nobs
    S_inv = np.linalg.inv(S)
    K = B_obs @ S_inv  # shape (nobs, nobs)
    
    # Update observed components
    update = K @ y_innov  # shape (nobs,)
    
    # Localized update: only update near observed points
    # For Lorenz-96, use exponential localization with range ~5
    result = ensemble.copy()
    for j, obs_idx in enumerate(observed_indices):
        for i in range(nx):
            # Periodic distance
            dist = min(abs(i - obs_idx), nx - abs(i - obs_idx))
            # Exponential localization
            if dist <= 5:
                loc_weight = np.exp(-0.5 * (dist / 3.0) ** 2)
            else:
                loc_weight = 0.0
            result[:, i] += loc_weight * update[j]
    
    # Adaptive inflation to prevent filter collapse
    # Compute innovation variance vs expected
    innov_var = np.var(y_innov)
    expected_innov_var = np.mean(np.diag(S))
    
    if expected_innov_var > 0:
        ratio = innov_var / expected_innov_var
        # Inflate if innovations are too small
        if ratio < 0.9:
            inflation = 1.0 + 0.01 * (0.9 - ratio)
            result = x_mean + inflation * (result - x_mean)
    
    # Ensure finite values and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
