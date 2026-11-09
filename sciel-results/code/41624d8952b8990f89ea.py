"""Stochastic Ensemble Kalman Filter (SE-Kalman) for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter update with adaptive spread control.
    
    Uses proper stochastic perturbation to preserve ensemble spread while
    matching the optimal Kalman gain. Includes adaptive inflation to handle
    chaotic system error growth.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each with 20 state variables.
    observations : ndarray, shape (10,)
        Noisy observations of the 10 even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_ens = ensemble.shape[0]  # 16
    n_state = ensemble.shape[1]  # 20
    n_obs = len(observed_indices)  # 10
    
    # Compute ensemble mean and anomalies
    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean  # shape (n_ens, n_state)
    
    # Project anomalies onto observed indices
    obs_anomalies = anomalies[:, observed_indices]  # shape (n_ens, n_obs)
    
    # Ensemble covariance matrices
    C_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)  # shape (n_obs, n_obs)
    C_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)      # shape (n_state, n_obs)
    
    # Innovation covariance
    R_obs = observation_variance * np.eye(n_obs)
    S = C_oo + R_obs
    
    # Innovation vector
    innovation = observations - ens_mean[observed_indices]  # shape (n_obs,)
    
    # Compute Kalman gain: K = C_xo @ S^{-1}
    try:
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(S)
        # Solve L @ L^T @ X^T = C_xo^T, then K = X
        temp = np.linalg.solve(L, C_xo.T)
        K_T = np.linalg.solve(L.T, temp)
        K = K_T.T  # shape (n_state, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C_xo @ S_inv
    
    # Compute mean update
    new_mean = ens_mean + K @ innovation
    
    # Adaptive inflation: estimate if ensemble spread is too low
    # Compare observed innovation variance to expected
    innovation_var = np.var(innovation)
    expected_innovation_var = np.trace(S) / n_obs
    
    # If innovations are larger than expected, inflate the ensemble
    inflation_factor = 1.0
    if expected_innovation_var > 1e-10:
        ratio = innovation_var / expected_innovation_var
        # Mild adaptive inflation
        if ratio > 1.5:
            inflation_factor = 1.0 + 0.1 * (ratio - 1.0)
        elif ratio < 0.5:
            inflation_factor = 1.0 - 0.05 * (1.0 - ratio)
    
    # Apply inflation to anomalies
    if inflation_factor != 1.0:
        anomalies = anomalies * inflation_factor
        obs_anomalies = obs_anomalies * inflation_factor
        # Recompute covariance and gain with inflated spread
        C_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)
        C_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)
        S = C_oo + R_obs
        try:
            L = np.linalg.cholesky(S)
            temp = np.linalg.solve(L, C_xo.T)
            K_T = np.linalg.solve(L.T, temp)
            K = K_T.T
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = C_xo @ S_inv
        new_mean = ens_mean + K @ innovation
    
    # Stochastic SE-Kalman update
    # Use random perturbations to preserve spread correctly
    # For each member: x_i' = x_i + K @ (y_i^pert - H x_i)
    # where y_i^pert = y + sqrt(R) * w_i
    
    sqrt_R = np.sqrt(observation_variance)
    
    # Use a fixed seed for reproducibility within each call
    rng = np.random.default_rng(42)
    
    updated_ensemble = np.zeros_like(ensemble)
    for i in range(n_ens):
        # Generate random perturbation for this member
        w_i = rng.standard_normal(n_obs)
        y_i_pert = observations + sqrt_R * w_i
        
        # Compute member-specific innovation
        H_x_i = ensemble[i, observed_indices]
        d_i = y_i_pert - H_x_i
        
        # Update member
        updated_ensemble[i, :] = ensemble[i, :] + K @ d_i
    
    # Correct the mean to ensure it matches the desired posterior mean exactly
    # The stochastic update may have a small bias in the mean
    current_mean = updated_ensemble.mean(axis=0)
    mean_correction = new_mean - current_mean
    updated_ensemble += mean_correction
    
    # Ensure output is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
