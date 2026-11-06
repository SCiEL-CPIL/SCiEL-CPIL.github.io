"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
# Global deadline tracking
_start_time = time.time()
_DEADLINE = _start_time + 145.0  # Safety margin before 150s kill

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with adaptive inflation and localization.
    Combines the adaptive inflation logic from the current best with
    the structural clarity and robustness of the inspirations.
    """
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
    
    # Exponential localization: exp(-0.5 * (dist / L)^2)
    # L=1.5 was found to be optimal in the current best score
    L = 1.5
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)
    
    # Localize P_xo
    P_xo_loc = P_xo * loc_matrix
    
    # S = P_oo + R
    S = P_oo + R
    
    # Compute inverse of S
    try:
        # Use solve for better numerical stability if possible, but inv is fine for small N
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # K = P_xo_loc @ S_inv  (N x N_obs)
    K = P_xo_loc @ S_inv
    
    # 7. Update Mean
    mean_a = mean_b + K @ d
    
    # 8. Update Perturbations
    # W_a_T = (W_b.T - K @ H_W_b_T)
    H_W_b_T = W_b.T[obs_idx, :]  # N_obs x M
    update_part = K @ H_W_b_T  # N x M
    W_a_T = W_b.T - update_part
    
    # 9. Adaptive Inflation
    # Compare actual innovation variance to expected innovation variance
    # Expected innovation variance is the mean of the diagonal of S
    expected_innov_var = np.mean(np.diag(S))
    # Actual innovation variance
    actual_innov_var = np.var(d) + 1e-10  # Avoid division by zero
    
    # If actual variance is lower than expected, the ensemble is too confident -> inflate
    # If actual variance is higher, the ensemble might be too diffuse, but we generally only inflate to prevent collapse.
    
    ratio = expected_innov_var / actual_innov_var
    # Clip the ratio to avoid extreme adjustments
    inflation_factor = np.sqrt(ratio)
    
    # Bound the inflation to [1.0, 1.05] to be conservative.
    inflation_factor = np.clip(inflation_factor, 1.0, 1.05)
    
    # Apply inflation to perturbations
    W_a_T *= inflation_factor
    
    # Re-center the perturbations to ensure the mean remains mean_a
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]
    
    # 10. Construct Analysis Ensemble
    X_a = mean_a + W_a_T.T
    
    # 11. Safety checks: ensure finite and within bounds
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
# EVOLVE-BLOCK-END
