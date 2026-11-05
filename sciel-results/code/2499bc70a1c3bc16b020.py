"""Lorenz-96 assimilation using a localized Ensemble Square Root Filter (EnSRF) with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x has shape (N, 20) or (20,)."""
    if x.ndim == 1:
        x_1 = np.roll(x, 1)
        x_m1 = np.roll(x, -1)
        x_m2 = np.roll(x, 2)
        return (x_1 - x_m2) * x_m1 - x + F
    else:
        x_1 = np.roll(x, 1, axis=1)
        x_m1 = np.roll(x, -1, axis=1)
        x_m2 = np.roll(x, 2, axis=1)
        return (x_1 - x_m2) * x_m1 - x + F


def _circular_distance(i, j, n):
    """Compute circular distance between indices i and j on a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Square Root Filter analysis update.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Make a copy to work with
    result = ensemble.copy()
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean
    
    # Compute sample covariance (N-1 normalization)
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation error covariance (diagonal)
    R = np.eye(n_obs) * observation_variance
    
    # Observation innovation
    innovation = observations - H @ ens_mean
    
    # Innovation covariance: S = H P H^T + R
    HP = H @ cov
    S = HP @ H.T + R
    
    # Kalman gain: K = P H^T S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = (cov @ H.T) @ S_inv  # Shape: (n_state, n_obs)
    
    # Apply localization to the Kalman gain
    # Use a Gaussian decay based on circular distance
    # Tighter localization for 20-variable system: ~1/4 of the ring
    decay_length = 1.8
    
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            # Circular distance
            d = _circular_distance(i, obs_idx, n_state)
            w = np.exp(-d**2 / (2 * decay_length**2))
            localized_K[i, j] = K[i, j] * w
    
    # Update each ensemble member using EnSRF approach
    # For each member, adjust based on the member-specific innovation
    for m in range(n_members):
        # Compute H x_m^f
        Hx_m = H @ result[m]
        
        # Innovation for this member
        innov_m = observations - Hx_m
        
        # Update
        result[m] = result[m] + localized_K @ innov_m
    
    # Adaptive inflation to prevent filter collapse
    # Use a smoother, more stable inflation scheme
    
    # Estimate expected innovation variance
    S_expected = HP @ H.T + R
    S_expected_diag = np.diag(S_expected)
    
    # Actual innovation variance (use per-observation for better stability)
    innov_var_per_obs = np.var(innovation)
    
    # Compute inflation factor using mean ratio for stability
    # If the ensemble spread is too small, inflate
    # Target: S_actual ≈ S_expected
    if np.any(S_expected_diag > 0):
        # Use geometric mean of ratios for stability
        valid_mask = S_expected_diag > 1e-10
        if np.any(valid_mask):
            ratios = innov_var_per_obs / S_expected_diag[valid_mask]
            ratios = np.clip(ratios, 0.5, 2.0)
            log_ratio = np.mean(np.log(ratios))
            inflation = np.exp(np.clip(0.5 * log_ratio, -0.15, 0.15))
        else:
            inflation = 1.0
    else:
        inflation = 1.0
    
    # Apply inflation around the analysis mean
    new_mean = np.mean(result, axis=0)
    result = new_mean + inflation * (result - new_mean)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

# EVOLVE-BLOCK-END
