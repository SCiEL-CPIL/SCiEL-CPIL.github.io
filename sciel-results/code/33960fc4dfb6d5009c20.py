"""Combined Global Deterministic ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(r, width=2.0):
    """Gaspari-Cohn compactly supported correlation function."""
    r = np.abs(r)
    if r <= width / 2.0:
        return (1.0 / 2.0) * (r / (width / 2.0)) ** 6 * (10.0 - 15.0 * (r / (width / 2.0)) ** 2 + 6.0 * (r / (width / 2.0)) ** 4)
    elif r <= width:
        return (2.0 / width) - r / (width ** 2) - (7.0 / (2.0 * width ** 3)) * r ** 3 + (12.0 / width ** 4) * r ** 4 - (6.0 / width ** 5) * r ** 5 + (5.0 / (2.0 * width ** 6)) * r ** 6
    else:
        return 0.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Global Deterministic Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure inputs are finite
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Build the full observation operator H: (K, N)
    H = np.zeros((K, N))
    for k, idx in enumerate(observed_indices):
        H[k, int(idx)] = 1.0
    
    # Ensemble mean
    x_mean = np.mean(ensemble, axis=0)
    
    # Anomalies
    A = ensemble - x_mean  # (M, N)
    
    # Full ensemble covariance C = A^T A / (M-1)
    C = (A.T @ A) / (M - 1)  # (N, N)
    
    # Innovation
    innovation = observations - H @ x_mean  # (K,)
    
    # Innovation covariance S = H C H^T + R
    HC = H @ C  # (K, N)
    S = HC @ H.T + observation_variance * np.eye(K)  # (K, K)
    
    # Compute Kalman Gain K_gain = C H^T S^{-1}
    # To avoid explicit inverse, solve S Z^T = H C^T
    try:
        HC_T = H @ C.T  # (K, N)
        Z_T = np.linalg.solve(S, HC_T)  # (K, N)
        K_gain = Z_T.T  # (N, K)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_gain = C @ H.T @ S_inv
    
    # Deterministic update for each member
    # x_m^a = x_m + K_gain (y - H x_m)
    updated_ens = np.zeros_like(ensemble)
    for m in range(M):
        Hx = H @ ensemble[m, :]
        updated_ens[m, :] = ensemble[m, :] + K_gain @ (observations - Hx)
    
    # Final safety checks
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ens
# EVOLVE-BLOCK-END
