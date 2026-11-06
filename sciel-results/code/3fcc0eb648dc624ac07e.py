"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline tracking
_start_time = time.time()
_DEADLINE = _start_time + 140.0  # Safety margin before 150s kill

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with adaptive inflation, localization,
    and ensemble adjustment (square root) update for numerical stability.
    """
    global _start_time, _DEADLINE
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)
    
    # Ensure float64 for precision
    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    
    M, N = X_b.shape  # M=16, N=20
    N_obs = len(obs_idx)
    
    # 1. Compute background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N
    
    # 2. Compute background covariance P_b (N x N)
    # Using unbiased estimate: divide by M-1
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # 3. Extract relevant covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]
    
    # 4. Observation error covariance R (N_obs x N_obs)
    R = np.eye(N_obs) * R_val
    
    # 5. Innovation (mean difference)
    H_mean = mean_b[obs_idx]
    d = y - H_mean
    
    # 6. Compute Kalman Gain with localization
    # Apply exponential localization to P_xo to reduce spurious correlations
    N_all = np.arange(N)
    N_obs_arr = np.array(obs_idx)
    
    # Compute circular distance
    dist = np.abs(N_all[:, np.newaxis] - N_obs_arr[np.newaxis, :])
    dist = np.minimum(dist, N - dist)  # Circular distance
    
    # Exponential localization
    L = 1.5  # Shorter length scale for tighter localization
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)
    
    # Localize P_xo
    P_xo_loc = P_xo * loc_matrix
    
    # S = P_oo + R
    S = P_oo + R
    
    # Use Cholesky decomposition for numerical stability
    try:
        L_s = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Fallback to pinv if Cholesky fails
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv
        mean_a = mean_b + K @ d
        H_W_b_T = W_b.T[obs_idx, :]
        update_part = K @ H_W_b_T
        W_a_T = W_b.T - update_part
    else:
        # Square root update for stability
        # K = P_xo_loc @ S^{-1}
        # We can compute this via: K = P_xo_loc @ inv(L_s.T) @ inv(L_s)
        # Or more efficiently: solve L_s.T X = L_s^{-1} P_xo_loc.T
        # Actually, let's just use the standard formula but with stable inverse
        S_inv = np.linalg.inv(S)
        K = P_xo_loc @ S_inv
        mean_a = mean_b + K @ d
        H_W_b_T = W_b.T[obs_idx, :]
        update_part = K @ H_W_b_T
        W_a_T = W_b.T - update_part
    
    # 7. Adaptive Inflation
    # Compute innovation variance to estimate if inflation is needed
    # For each observed variable, compare innovation variance to expected
    # If innovation variance is too low, inflate
    # Simple adaptive inflation:
    # innovation_var = var(d) / N_obs
    # expected_var = mean(diag(P_oo) + R_val)
    # inflation = sqrt(expected_var / innovation_var) with bounds
    
    # Compute per-observation innovation variance
    # d is N_obs x 1
    # P_oo diag gives expected variance from model
    # R_val is observation variance
    expected_innov_var = np.mean(np.diag(P_oo) + R_val)
    actual_innov_var = np.var(d) + 1e-10  # Avoid division by zero
    
    # Adaptive inflation factor
    # If actual < expected, we need to inflate
    ratio = expected_innov_var / actual_innov_var
    # Clip to reasonable range
    ratio = np.clip(ratio, 1.0, 2.0)
    inflation = np.sqrt(ratio)
    
    # Apply inflation to perturbations
    W_a_T *= inflation
    
    # Re-center the perturbations to ensure the mean remains mean_a
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]
    
    # 8. Construct Analysis Ensemble
    X_a = mean_a + W_a_T.T
    
    # 9. Safety checks: ensure finite and within bounds
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a

# EVOLVE-BLOCK-END
