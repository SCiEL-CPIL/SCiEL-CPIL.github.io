import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and adaptive inflation
    for Lorenz-96 partial-observation assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states.
    observations: (K,) array of noisy observations for the observed indices.
    observed_indices: (K,) array of indices that are observed.
    observation_variance: scalar variance of the observation noise.
    
    Returns:
    updated_ensemble: (M, N) array of updated ensemble states.
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Compute ensemble covariance with localization
    # P_f = (1/(M-1)) * anomalies.T @ anomalies
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply Gaspari-Cohn localization
    # Use a correlation length scale of 4.0 for N=20
    L = 4.0
    def gaspari_cohn(d, L):
        if d <= L/2:
            return (-0.25 * (2*d/L - 5)**5 + 5*(2*d/L - 3)**5 + 5*(2*d/L - 1)**5 + 5) / 4
        elif d <= L:
            return (0.5 * (2*d/L - 7)**5 - 5*(2*d/L - 5)**5 + 10*(2*d/L - 3)**5 - 5*(2*d/L - 1)**5 + 5) / 2
        else:
            return 0.0
    
    # Vectorized localization
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)
    
    # Compute localization matrix
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            loc_matrix[i, j] = gaspari_cohn(dist[i, j], L)
    
    P_f *= loc_matrix
    
    # 4. Observation operator H is implicit via indices
    # S = H P_f H^T + R
    # R is K x K diagonal with observation_variance
    H_P_H = P_f[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # 5. Kalman Gain K_g = P_f H^T S^-1
    # We need P_f[:, observed_indices] @ inv(S)
    # P_f_obs = P_f[:, observed_indices] (N x K)
    P_f_obs = P_f[:, observed_indices]
    
    # Solve S X = P_f_obs^T to get X = S^-1 P_f_obs^T
    # Then K_g = X^T (K x N) -> actually K_g = P_f_obs @ S^-1 (N x K)
    try:
        # Use Cholesky for stability if possible
        S_cholesky = np.linalg.cholesky(S)
        S_inv_P_f_obs_T = np.linalg.solve(S_cholesky.T, np.linalg.solve(S_cholesky, P_f_obs.T))
        K_g = S_inv_P_f_obs_T.T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        S_inv = np.linalg.pinv(S)
        K_g = P_f_obs @ S_inv
    
    # 6. Innovation
    obs_mean = mean_f[observed_indices]
    innovation = obs - obs_mean
    
    # 7. Update mean
    mean_a = mean_f + K_g @ innovation
    
    # 8. Update anomalies
    # Efficient update:
    # x_m^a = mean_a + (x_m^f - mean_f) @ (I - K_g H)
    # Let A = anomalies (M x N)
    # We want A @ (I - H^T K_g^T) ? No.
    # x_m^a = mean_a + (x_m^f - mean_f) - (x_m^f - mean_f) H^T K_g^T
    # (x_m^f - mean_f) is row vector 1 x N.
    # H^T is N x K.
    # K_g^T is K x N.
    # So (x_m^f - mean_f) @ H^T @ K_g^T is 1 x N.
    
    # Let's compute the correction term for each member.
    # anomalies are M x N.
    # We need to subtract anomalies @ H^T @ K_g^T
    # H^T @ K_g^T is N x N.
    # Let M_update = I - H^T @ K_g^T
    # Then anomalies_a = anomalies @ M_update
    
    # Construct H^T K_g^T
    # H^T is N x K. K_g is N x K. K_g^T is K x N.
    # H^T @ K_g^T is N x N.
    H_T_K_g_T = np.zeros((N, N))
    for k, idx in enumerate(observed_indices):
        H_T_K_g_T[idx, :] = K_g[:, k]
        
    M_update = np.eye(N) - H_T_K_g_T
    anomalies_a = anomalies @ M_update
    
    # 9. Adaptive Inflation
    # Compute the innovation variance
    # The expected innovation variance is trace(S)/K
    # The actual innovation variance is mean(innovation^2)
    # If actual > expected, we need to inflate
    
    # For adaptive inflation, we adjust the ensemble spread
    # A simple approach: if the ensemble is too confident, inflate
    
    # Compute the ratio of actual to expected innovation variance
    actual_innov_var = np.mean(innovation ** 2)
    expected_innov_var = np.trace(S) / K
    
    # If actual > expected * 1.1, inflate
    inflation_factor = 1.0
    if expected_innov_var > 1e-10:
        ratio = actual_innov_var / expected_innov_var
        if ratio > 1.1:
            # Inflate by a factor that brings the ratio closer to 1
            # inflation_factor = sqrt(ratio) * 1.05 (with a cap)
            inflation_factor = min(np.sqrt(ratio) * 1.02, 1.2)
    
    # Apply inflation to anomalies
    anomalies_a *= inflation_factor
    
    # 10. Reconstruct ensemble
    ensemble_a = mean_a[None, :] + anomalies_a
    
    # Ensure finite and within bounds
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
    
    return ensemble_a
