import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with Localization and Variance-Preserving Inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    M, N = ensemble.shape
    if M < 2:
        return ensemble.copy()
        
    # 1. Compute ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    X = ensemble - x_mean
    
    # 2. Compute ensemble covariance
    P = (X @ X.T) / (M - 1)
    
    # 3. Localization
    # Use a cosine taper with radius L=4
    L = 4
    # Create distance matrix
    idx = np.arange(N)
    dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
    
    # Periodic distance (Lorenz-96 is periodic)
    dist = np.minimum(dist, N - dist)
    
    # Taper function
    # w = 1 - 0.5 * (1 - cos(pi * d / L)) for d <= L, else 0
    W = np.zeros((N, N))
    mask = dist <= L
    W[mask] = 1 - 0.5 * (1 - np.cos(np.pi * dist[mask] / L))
    
    P_loc = P * W
    
    # 4. Observation Operator H
    # H is a sparse matrix, but we can construct it explicitly for small N
    H = np.zeros((len(observed_indices), N))
    for i, obs_idx in enumerate(observed_indices):
        H[i, obs_idx] = 1.0
        
    # 5. Observation Error Covariance R
    R = observation_variance * np.eye(len(observed_indices))
    
    # 6. Kalman Gain
    # S = H P_loc H^T + R
    HP = H @ P_loc
    S = HP @ H.T + R
    S_inv = np.linalg.inv(S)
    
    K = HP.T @ S_inv
    
    # 7. Update Mean
    innovation = observations - (H @ x_mean)
    x_mean_new = x_mean + K @ innovation
    
    # 8. Update Anomalies
    # X_new = (I - K H) X
    I = np.eye(N)
    KH = K @ H
    X_new = (I - KH) @ X
    
    # 9. Variance Preservation
    # Compute trace of localized covariance before and after
    tr_old = np.trace(P_loc)
    P_new = (X_new @ X_new.T) / (M - 1)
    tr_new = np.trace(P_new)
    
    if tr_new > 1e-10:
        alpha = np.sqrt(tr_old / tr_new)
    else:
        alpha = 1.0
        
    # Optional: Clip alpha to prevent extreme inflation/deflation
    alpha = np.clip(alpha, 0.5, 2.0)
    
    X_final = alpha * X_new
    
    # 10. Reconstruct Ensemble
    result = x_mean_new + X_final
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        # Fallback to simple nudging if something went wrong
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        
    # Clip to [-100, 100] as per task requirements
    result = np.clip(result, -100, 100)
    
    return result
