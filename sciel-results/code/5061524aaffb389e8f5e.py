"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with:
    - Gaussian localization of cross-covariance
    - Adaptive multiplicative inflation
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = observation_variance
    obs_idx = np.asarray(observed_indices, dtype=int)
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    # Copy ensemble
    result = np.array(ensemble, dtype=np.float64).copy()
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean  # (M, N)
    
    # --- Observation innovation ---
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 1:
        obs_arr = np.tile(obs_arr, (M, 1))
    
    H_ens = result[:, obs_idx]  # (M, n_obs)
    innovation = obs_arr - H_ens  # (M, n_obs)
    
    n_obs = len(obs_idx)
    
    # Anomalies for observed variables: (M, n_obs)
    anom_obs = anomalies[:, obs_idx]  # (M, n_obs)
    
    # --- Compute cross-covariance with Gaussian localization ---
    # Cross-covariance: (N x n_obs) = (1/(M-1)) * anomalies^T @ anom_obs
    cross_cov = (anomalies.T @ anom_obs) / (M - 1)  # (N, n_obs)
    
    # Build Gaussian localization weights
    L_loc = 5.0  # localization length scale
    loc_weights = np.zeros((N, n_obs))
    for j_idx, j in enumerate(obs_idx):
        for i in range(N):
            d = abs(i - j)
            d = min(d, N - d)  # periodic distance
            loc_weights[i, j_idx] = np.exp(-0.5 * (d / L_loc) ** 2)
    
    # Apply localization
    cross_cov_loc = cross_cov * loc_weights
    
    # Innovation covariance: (n_obs x n_obs) = (1/(M-1)) * anom_obs^T @ anom_obs + R
    innov_cov = (anom_obs.T @ anom_obs) / (M - 1) + obs_var * np.eye(n_obs)
    
    # Invert innovation covariance
    try:
        innov_cov_inv = np.linalg.inv(innov_cov)
    except np.linalg.LinAlgError:
        innov_cov_inv = np.linalg.pinv(innov_cov)
    
    # Kalman gain: K = cross_cov_loc @ innov_cov_inv  (N x n_obs)
    K = cross_cov_loc @ innov_cov_inv
    
    # Apply update
    update = (K @ innovation.T).T  # (M, N)
    result = result + update
    
    # --- Adaptive Inflation ---
    # Compute ensemble spread for observed variables
    ens_spread_obs = np.std(result[:, obs_idx], axis=0, ddof=1)  # (n_obs,)
    
    # Expected spread from observation noise
    expected_spread = np.sqrt(obs_var)
    
    # Inflation factor
    min_spread = 0.3 * expected_spread
    inflation_factors = np.ones(n_obs)
    for j in range(n_obs):
        if ens_spread_obs[j] < min_spread:
            inflation_factors[j] = min_spread / max(ens_spread_obs[j], 1e-10)
            inflation_factors[j] = min(inflation_factors[j], 1.5)  # cap
    
    # Apply inflation to all variables
    avg_inflation = np.mean(inflation_factors)
    if avg_inflation > 1.0:
        result = ens_mean + avg_inflation * (result - ens_mean)
    
    # Safety: clip to valid range
    np.clip(result, -100.0, 100.0, out=result)
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.array(ensemble, dtype=np.float64).copy()
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
    
    return result

# EVOLVE-BLOCK-END
