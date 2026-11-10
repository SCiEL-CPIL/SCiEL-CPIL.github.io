"""Ensemble Kalman Filter analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with adaptive inflation and localization.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states.
    observations: (K,) or (1, K) array of noisy observations for observed indices.
    observed_indices: array of indices that are observed.
    observation_variance: scalar variance of the observation noise.
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Adaptive inflation based on innovation magnitude
    # Compute innovation norm to gauge filter confidence
    obs_mean_f = mean_f[observed_indices]
    innovation = obs - obs_mean_f
    innovation_norm = np.linalg.norm(innovation)
    
    # Base inflation with adaptive adjustment
    # If innovations are large, inflate more to prevent underdispersion
    base_inflation = 1.05
    # Scale inflation by how much the ensemble spread agrees with observation noise
    # Compute expected innovation variance from ensemble
    P_f_raw = (anomalies.T @ anomalies) / (M - 1)
    expected_innov_var = np.diag(P_f_raw[np.ix_(observed_indices, observed_indices)])
    expected_innov_norm = np.sqrt(np.sum(expected_innov_var + K * observation_variance))
    
    # Adaptive inflation: increase if observed innovation exceeds expected
    if expected_innov_norm > 1e-10:
        ratio = innovation_norm / expected_innov_norm
        # Smoothly increase inflation for large ratios
        adaptive_boost = 0.03 * np.tanh(ratio - 1.0)
    else:
        adaptive_boost = 0.0
    
    inflation_factor = base_inflation + adaptive_boost
    # Cap inflation to avoid excessive spread
    inflation_factor = min(inflation_factor, 1.15)
    
    anomalies *= inflation_factor
    
    # 4. Compute ensemble covariance with localization
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply Gaspari-Cohn localization
    L = 4.0  # Correlation length scale in indices
    
    # Build localization matrix efficiently
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            if d <= L:
                if d <= L / 2:
                    x1 = 2 * d / L - 5
                    x2 = 2 * d / L - 3
                    x3 = 2 * d / L - 1
                    loc_matrix[i, j] = (-0.25 * x1**5 + 5 * x2**5 + 5 * x3**5 + 5) / 4
                else:
                    x1 = 2 * d / L - 7
                    x2 = 2 * d / L - 5
                    x3 = 2 * d / L - 3
                    x4 = 2 * d / L - 1
                    loc_matrix[i, j] = (0.5 * x1**5 - 5 * x2**5 + 10 * x3**5 - 5 * x4**5 + 5) / 2
            else:
                loc_matrix[i, j] = 0.0
    
    P_f_loc = P_f * loc_matrix
    
    # 5. Observation operator H is implicit via indices
    # S = H P_f H^T + R
    H_P_H = P_f_loc[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # Regularize S for numerical stability
    S += 1e-8 * np.eye(K)
    
    # 6. Kalman Gain K_g = P_f H^T S^-1
    P_f_obs = P_f_loc[:, observed_indices]
    
    try:
        S_inv_P_f_obs_T = np.linalg.solve(S, P_f_obs.T)
        K_g = S_inv_P_f_obs_T.T  # Shape: N x K
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_g = P_f_obs @ S_inv  # Shape: N x K
    
    # 7. Innovation
    innovation = obs - obs_mean_f
    
    # 8. Update mean
    mean_a = mean_f + K_g @ innovation
    
    # 9. Update anomalies using deterministic update
    # Construct H^T K_g^T matrix (N x N)
    H_T_K_g_T = np.zeros((N, N))
    for k, idx in enumerate(observed_indices):
        H_T_K_g_T[idx, :] = K_g[:, k]
        
    M_update = np.eye(N) - H_T_K_g_T
    anomalies_a = anomalies @ M_update
    
    # 10. Reconstruct ensemble
    ensemble_a = mean_a[None, :] + anomalies_a
    
    # Ensure finite and within bounds
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
    
    return ensemble_a
# EVOLVE-BLOCK-END
