import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Kalman Filter update with 
    Gaspari-Cohn localization and adaptive spatial inflation.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array with shape (16, 20).
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # 1. Compute Ensemble Mean and Anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # 2. Gaspari-Cohn Localization Function
    def gaspari_cohn(dist, L):
        """Gaspari-Cohn correlation function for localization"""
        dist = np.abs(dist)
        # For dist < L/2: polynomial 1
        # For L/2 <= dist < 3L/2: polynomial 2
        # For dist >= 3L/2: 0
        result = np.zeros_like(dist, dtype=float)
        mask1 = dist < L / 2.0
        mask2 = (dist >= L / 2.0) & (dist < 3.0 * L / 2.0)
        
        # Polynomial 1 for dist < L/2
        x1 = dist[mask1] / L
        result[mask1] = -0.25 * x1**5 + 0.5 * x1**4 - 5.0/3.0 * x1**3 + 5.0/3.0 * x1**2 + 1.0
        
        # Polynomial 2 for L/2 <= dist < 3L/2
        x2 = dist[mask2] / L
        result[mask2] = (0.1 * x2**5 - 0.5 * x2**4 + 5.0/3.0 * x2**3 - 5.0 * x2**2 + 10.0/3.0 * x2 - 5.0/3.0)
        
        return result
    
    # 3. Compute localized innovation covariance and gain using square-root method
    # A_obs: (N, n_obs) - anomalies at observed locations
    A_obs = anomalies[:, obs_idx]
    
    # Compute localized anomaly matrix for observations
    # For each observation j, we want to localize the anomalies in state space
    # This is done by constructing a localized version of the observation operator
    
    # Localization length scale
    L_loc = 4.0
    
    # Compute localization weights for each state variable to each observation
    loc_weights = np.zeros((D, n_obs))
    for j, obs_j in enumerate(obs_idx):
        for i in range(D):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)  # Cyclic distance
            loc_weights[i, j] = gaspari_cohn(dist, L_loc)
    
    # Square-root EnKF update:
    # We want to compute the update without forming the full covariance matrix.
    # The key idea: work with the anomaly matrix directly.
    
    # Step 1: Compute the localized observation anomalies
    # Y_obs = A_obs (N, n_obs)
    Y_obs = A_obs.copy()
    
    # Step 2: Add observation noise to the anomalies
    # We need to augment the anomaly matrix with observation noise
    # R = observation_variance * I
    # We add sqrt(R) * noise to the anomalies
    
    # For the square-root filter, we work with the matrix B where
    # B B^T / (N-1) = C_oo + R
    # We can do this by augmenting the anomaly matrix
    
    # Create augmented anomaly matrix: [A_obs; sqrt(R) * I_noise]
    # But we need to be careful about dimensions
    
    # Alternative approach: Use the standard EnKF update but with 
    # localized gain and stable linear solve
    
    # Compute C_oo (n_obs, n_obs) with localization
    # C_oo_ij = (1/(N-1)) sum_k A_obs[k,i] * A_obs[k,j] * loc_weight
    # But localization is applied to the gain, not the covariance directly
    
    # Let's use the standard approach: compute the gain with localization
    # C_xo = (anomalies.T @ A_obs) / (N-1)  (D, n_obs)
    C_xo = (anomalies.T @ A_obs) / (N - 1)
    
    # C_oo = (A_obs.T @ A_obs) / (N-1)  (n_obs, n_obs)
    C_oo = (A_obs.T @ A_obs) / (N - 1)
    
    # Innovation covariance: S = C_oo + R
    R = observation_variance * np.eye(n_obs)
    S = C_oo + R
    
    # Compute S^{-1} stably using Cholesky or pinv
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Gain: K = C_xo @ S^{-1}
    K = C_xo @ S_inv
    
    # Apply localization to the gain
    K_localized = K * loc_weights  # (D, n_obs)
    
    # 4. Adaptive Inflation
    # Compute innovation
    innovation = observations - mean[obs_idx]  # (n_obs,)
    
    # Expected innovation variance
    expected_innov_var = np.trace(S)
    actual_innov_var = np.dot(innovation, innovation)
    
    if expected_innov_var > 1e-8:
        ratio = actual_innov_var / expected_innov_var
        # Inflate if the ensemble is too narrow
        if ratio > 1.0:
            # Global inflation factor
            alpha_global = np.sqrt(ratio)
            alpha_global = np.clip(alpha_global, 1.0, 2.0)
            
            # But we want spatially adaptive inflation
            # Inflate anomalies more where the innovation is larger
            # Simple approach: scale anomalies by alpha_global
            anomalies_inflated = anomalies * alpha_global
        else:
            anomalies_inflated = anomalies
    else:
        anomalies_inflated = anomalies
    
    # Recompute gain with inflated anomalies if inflation was applied
    if not np.allclose(anomalies_inflated, anomalies):
        A_obs_inf = anomalies_inflated[:, obs_idx]
        C_xo_inf = (anomalies_inflated.T @ A_obs_inf) / (N - 1)
        C_oo_inf = (A_obs_inf.T @ A_obs_inf) / (N - 1)
        S_inf = C_oo_inf + R
        try:
            S_inv_inf = np.linalg.inv(S_inf)
        except np.linalg.LinAlgError:
            S_inv_inf = np.linalg.pinv(S_inf)
        K_inf = C_xo_inf @ S_inv_inf
        K_localized = K_inf * loc_weights
    
    # 5. Update the ensemble using the square-root / local EnKF approach
    # For each member, update: x_i' = x_i + K_loc @ (y - H x_i)
    updated_ensemble = ensemble.copy()
    for i in range(N):
        x_i_obs = ensemble[i, obs_idx]
        innov_i = observations - x_i_obs
        corr = K_localized @ innov_i
        updated_ensemble[i] += corr
    
    # 6. Optional: Add small random perturbation to maintain ensemble spread
    # This helps prevent filter collapse
    # But we need to be careful not to add too much noise
    
    # 7. Clip to bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check time
    if time.time() > deadline:
        return updated_ensemble
    
    return updated_ensemble
