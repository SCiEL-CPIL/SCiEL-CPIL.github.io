"""Lorenz-96 assimilation using Vectorized Ensemble Square Root Filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Vectorized Ensemble Square Root Filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    ens = ensemble.astype(np.float64).copy()
    obs_idx = np.asarray(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean # (n_members, n_state)
    
    # 2. Construct H (observation operator)
    # H is (n_obs, n_state)
    H = np.zeros((n_obs, n_state))
    for j, oidx in enumerate(obs_idx):
        H[j, oidx] = 1.0
        
    # 3. Compute Covariances
    # C_xx = (1/(n-1)) * A^T A
    # A is (n_members, n_state)
    # C_xx is (n_state, n_state)
    scale = 1.0 / (n_members - 1)
    C_xx = scale * (anomalies.T @ anomalies)
    
    # C_yy = H C_xx H^T
    # C_yy is (n_obs, n_obs)
    C_yy = H @ C_xx @ H.T
    
    # C_xy = C_xx H^T
    # C_xy is (n_state, n_obs)
    C_xy = C_xx @ H.T
    
    # 4. Compute Kalman Gain
    # R is observation covariance (n_obs, n_obs)
    R = observation_variance * np.eye(n_obs)
    
    # S = C_yy + R
    S = C_yy + R
    
    # K = C_xy S^{-1}
    # Solve S K^T = C_xy^T  =>  K^T = S^{-1} C_xy^T
    try:
        # Using solve for stability
        K_T = np.linalg.solve(S, C_xy.T)
        K = K_T.T # (n_state, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        K = C_xy @ np.linalg.pinv(S)
        
    # 5. Update Mean
    # Innovation = y - H m_f
    innovations = observations - (H @ ens_mean)
    ens_mean_new = ens_mean + K @ innovations
    
    # 6. Update Anomalies
    # A_a = A (I - K H)
    # I is (n_state, n_state)
    I = np.eye(n_state)
    M = I - K @ H
    anomalies_new = anomalies @ M
    
    # 7. Reconstruct Ensemble
    ens_new = ens_mean_new + anomalies_new
    
    # 8. Adaptive Inflation
    # Check spread at observed points
    ens_obs_new = ens_new[:, obs_idx]
    ens_var_obs = np.var(ens_obs_new, axis=0, ddof=1)
    
    # Avoid division by zero
    ens_var_obs = np.maximum(ens_var_obs, 1e-10)
    
    # Target variance is observation variance
    # If ensemble spread is too low, inflate
    ratios = observation_variance / ens_var_obs
    mean_ratio = np.mean(ratios)
    
    # Inflate if mean_ratio > 1
    # Use a damping factor to avoid over-correction
    if mean_ratio > 1.0:
        inflation_factor = np.sqrt(mean_ratio)
        # Limit inflation to prevent instability
        inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
        
        # Apply inflation around the new mean
        ens_new = ens_mean_new + inflation_factor * (ens_new - ens_mean_new)
        
    # 9. Safety checks
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.nan_to_num(ens_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens_new

# EVOLVE-BLOCK-END
