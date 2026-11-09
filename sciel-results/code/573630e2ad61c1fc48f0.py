import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : np.ndarray, shape (16, 20)
        Current ensemble members.
    observations : np.ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : np.ndarray or list
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : np.ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    n_members = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Convert to numpy arrays if needed
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    n_obs = len(observed_indices)
    
    # --- Localization: use a simple exponential decay or fixed radius ---
    # For Lorenz-96 with 20 variables, a moderate localization radius works well.
    # We'll use a simple approach: only update observed coordinates directly,
    # and spread the update to nearby coordinates using a localization kernel.
    
    # Compute ensemble mean and covariance
    ensemble_mean = np.mean(ensemble, axis=0)
    
    # Anomalous matrix
    anomalies = ensemble - ensemble_mean  # shape (n_members, n_state)
    
    # Sample covariance
    # C = anomalies.T @ anomalies / (n_members - 1)
    # But we'll use the Ensemble Kalman Filter formulation (localization-friendly)
    
    # --- Ensemble Kalman Filter (perturbed observations method or gain matrix) ---
    # Using the gain matrix approach with localization
    
    # Observation operator H: picks out observed indices
    # H is n_obs x n_state
    
    # For localization, we'll use a simple approach:
    # 1. Compute the innovation
    # 2. Compute the Kalman gain with localized covariance
    
    # Localized covariance: use a simple exponential kernel
    # K(i,j) = exp(-|i-j| / L) where L is the localization length
    
    localization_length = 4.0  # moderate localization for L96
    
    # Compute the background covariance at observed indices
    # C_oo = anomalies[:, observed_indices].T @ anomalies[:, observed_indices] / (n_members - 1)
    
    obs_anomalies = anomalies[:, observed_indices]  # (n_members, n_obs)
    
    # Sample covariance of observations
    C_oo = (obs_anomalies.T @ obs_anomalies) / (n_members - 1)  # (n_obs, n_obs)
    
    # Add observation noise
    C_oo += observation_variance * np.eye(n_obs)
    
    # Innovation
    innovation = observations - ensemble_mean[observed_indices]  # (n_obs,)
    
    # Cross-covariance between state and observations (with localization)
    # C_xo[i, j] = Cov(x_i, y_j)
    # For each state i and observed j, compute the localized cross-covariance
    
    C_xo = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for j_idx, j in enumerate(observed_indices):
            # Localized cross-covariance
            loc_weight = np.exp(-abs(i - j) / localization_length)
            cross_cov = np.mean(anomalies[:, i] * obs_anomalies[:, j_idx])
            C_xo[i, j_idx] = loc_weight * cross_cov
    
    # Kalman gain: K = C_xo @ inv(C_oo)
    try:
        K = C_xo @ np.linalg.inv(C_oo)  # (n_state, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to simpler update if inversion fails
        K = np.zeros((n_state, n_obs))
        for i in range(n_state):
            for j_idx, j in enumerate(observed_indices):
                loc_weight = np.exp(-abs(i - j) / localization_length)
                C_xo_val = C_xo[i, j_idx]
                C_oo_val = C_oo[j_idx, j_idx]
                K[i, j_idx] = loc_weight * C_xo_val / C_oo_val if C_oo_val > 1e-10 else 0.0
    
    # Update ensemble
    updated_ensemble = ensemble.copy()
    for i in range(n_state):
        for m in range(n_members):
            updated_ensemble[m, i] = ensemble[m, i] + np.dot(K[i, :], innovation)
    
    # --- Adaptive Inflation ---
    # Check if the ensemble spread is too small relative to the innovation
    # Compute the ensemble spread at observed locations
    obs_spread = np.std(obs_anomalies, axis=0)  # (n_obs,)
    obs_innovation_std = np.std(innovation)
    
    # If the innovation is large relative to the spread, inflate the ensemble
    # Simple adaptive inflation: if innovation > spread, inflate
    inflation_factor = 1.0
    spread_mean = np.mean(obs_spread)
    if spread_mean < 1e-6:
        inflation_factor = 1.1
    elif obs_innovation_std > 2.0 * spread_mean:
        inflation_factor = 1.05
    
    # Apply inflation around the new mean
    new_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = new_mean + inflation_factor * (updated_ensemble - new_mean)
    
    # Ensure finiteness and bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
