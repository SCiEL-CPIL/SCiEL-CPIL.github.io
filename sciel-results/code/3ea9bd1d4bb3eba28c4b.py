import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF-A) with localization and adaptive inflation.
    Correctly updates both the ensemble mean and the ensemble spread.
    """
    n_members = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Convert to numpy arrays if needed
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    n_obs = len(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    ensemble_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ensemble_mean  # shape (n_members, n_state)
    
    # 2. Compute background covariance and cross-covariance
    # C_bb is n_state x n_state
    # C_bo is n_state x n_obs
    # C_oo is n_obs x n_obs
    
    # We only need C_bo and C_oo for the gain calculation
    obs_anomalies = anomalies[:, observed_indices]  # (n_members, n_obs)
    
    # Cross covariance C_bo: Cov(x, y)
    # C_bo[i, j] = Cov(x_i, y_j)
    C_bo = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for j_idx, j in enumerate(observed_indices):
            C_bo[i, j_idx] = np.mean(anomalies[:, i] * obs_anomalies[:, j_idx])
            
    # Observation covariance C_oo: Cov(y, y)
    C_oo = (obs_anomalies.T @ obs_anomalies) / (n_members - 1)
    
    # 3. Localization
    # Use exponential localization kernel
    localization_length = 3.0  # Tuned for L96 with 20 vars
    
    # Localize C_bo
    C_bo_loc = np.zeros_like(C_bo)
    for i in range(n_state):
        for j_idx, j in enumerate(observed_indices):
            dist = abs(i - j)
            weight = np.exp(-dist / localization_length)
            C_bo_loc[i, j_idx] = weight * C_bo[i, j_idx]
            
    # Localize C_oo
    C_oo_loc = np.zeros((n_obs, n_obs))
    for j1 in range(n_obs):
        for j2 in range(n_obs):
            dist = abs(observed_indices[j1] - observed_indices[j2])
            weight = np.exp(-dist / localization_length)
            C_oo_loc[j1, j2] = weight * C_oo[j1, j2]
            
    # Add observation noise to C_oo
    C_oo_loc += observation_variance * np.eye(n_obs)
    
    # 4. Compute Kalman Gain
    # K = C_bo_loc @ inv(C_oo_loc)
    try:
        # Use solve for stability
        K = np.linalg.solve(C_oo_loc.T, C_bo_loc.T).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        K = C_bo_loc @ np.linalg.pinv(C_oo_loc)
        
    # 5. Update Ensemble Mean
    innovation = observations - ensemble_mean[observed_indices]
    updated_mean = ensemble_mean + K @ innovation
    
    # 6. Update Anomalies (Spread)
    # X'_a = X'_b @ (I - K @ H)
    # H is the observation operator (selection matrix)
    # K @ H is (n_state, n_state)
    # We can compute (I - K @ H) @ anomalies.T ?
    # anomalies is (n_members, n_state).
    # X'_b is (n_members, n_state).
    # We want X'_a[m, i] = sum_j X'_b[m, j] * (I - K H)[j, i]
    # (I - K H) is (n_state, n_state).
    
    # Compute K @ H
    # H is sparse: H[j, i] = 1 if i == observed_indices[j]
    # K @ H: (n_state, n_obs) @ (n_obs, n_state)
    # Let's construct K_H
    K_H = np.zeros((n_state, n_state))
    for i in range(n_state):
        for j_idx, j in enumerate(observed_indices):
            K_H[i, j] = K[i, j_idx]
            
    # Update anomalies
    # X'_a = X'_b @ (I - K_H).T ?
    # Let's check dimensions.
    # X'_b: (N_m, N_s)
    # (I - K_H): (N_s, N_s)
    # We want X'_a[m, i] = sum_j X'_b[m, j] (delta_{ji} - (K H)_{ji})
    # This is X'_b @ (I - K_H).T if we view rows of X'_b as vectors?
    # Standard matrix multiplication: (A B)_{ij} = sum_k A_{ik} B_{kj}
    # We want (X'_a)_{m, i} = sum_j (X'_b)_{m, j} (I - K_H)_{j, i}
    # This is exactly X'_b @ (I - K_H).
    
    I = np.eye(n_state)
    update_matrix = I - K_H
    updated_anomalies = anomalies @ update_matrix
    
    # 7. Reconstruct Ensemble
    updated_ensemble = updated_mean + updated_anomalies
    
    # 8. Adaptive Inflation
    # Check if ensemble spread is too small at observed locations
    obs_spread = np.std(updated_anomalies[:, observed_indices], axis=0)
    obs_innovation_std = np.std(innovation)
    
    inflation_factor = 1.0
    spread_mean = np.mean(obs_spread)
    
    # If spread is very small, inflate significantly to avoid collapse
    if spread_mean < 1e-6:
        inflation_factor = 1.2
    elif obs_innovation_std > 2.0 * spread_mean:
        inflation_factor = 1.05
        
    # Apply inflation around the new mean
    updated_ensemble = updated_mean + inflation_factor * (updated_ensemble - updated_mean)
    
    # Ensure finiteness and bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
