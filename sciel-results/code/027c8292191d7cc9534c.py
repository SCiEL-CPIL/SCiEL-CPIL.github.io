"""Lorenz-96 ensemble assimilation: Deterministic EnKF with adaptive multiplicative inflation and Gaussian localization."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.05
_inflation_alpha = 0.85
_inflation_min = 1.0
_inflation_max = 2.5
_loc_range = 2.8

# Precompute localization weights
_loc_cache = {}

def _get_loc_weights(obs_idx, dim=20, rng=2.8):
    key = (obs_idx, dim, rng)
    if key not in _loc_cache:
        w = np.empty(dim)
        for j in range(dim):
            d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            w[j] = np.exp(-0.5 * (d / rng) ** 2)
        _loc_cache[key] = w
    return _loc_cache[key]

def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation
    t0 = time.time()
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # Observation operator
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # --- 1. Compute Mean and Anomalies ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (N, dim)
    
    # --- 2. Adaptive Inflation ---
    # Use forecast anomalies for innovation calculation (standard)
    y_f_mean = H @ mean
    innovation = observations - y_f_mean  # (n_obs,)
    
    # Covariance of forecast ensemble
    cov = np.cov(ensemble, rowvar=False)
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance S = H P H^T + R
    P_obs = cov[np.ix_(observed_indices, observed_indices)]
    S = P_obs + R
    S += 1e-10 * np.eye(n_obs)
    
    # Calculate innovation norm and expected norm
    innovation_norm_sq = innovation @ innovation
    expected_norm_sq = np.trace(S)
    
    # Adaptive inflation update
    # Target inflation is proportional to sqrt(innovation_norm / expected_norm)
    if expected_norm_sq > 1e-15:
        ratio = np.sqrt(innovation_norm_sq / expected_norm_sq)
        # If ratio > 1, we need more inflation
        target_inflation = max(1.0, min(ratio, _inflation_max))
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target_inflation
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # --- 3. Inflate Anomalies for Gain Calculation ---
    # We compute the gain based on the inflated covariance to trust observations less if spread is small
    inflated_anomalies = _inflation * anomalies
    inflated_cov = np.cov(ensemble + (inflated_anomalies - anomalies), rowvar=False) 
    # Note: cov(ensemble) is same as cov(anomalies). 
    # cov(inflated_ensemble) = _inflation^2 * cov(ensemble)
    # So we can just scale the cov
    P_inf = (_inflation ** 2) * cov
    
    # --- 4. Compute Kalman Gain ---
    # S_inf = H P_inf H^T + R
    P_obs_inf = P_inf[np.ix_(observed_indices, observed_indices)]
    S_inf = P_obs_inf + R
    S_inf += 1e-10 * np.eye(n_obs)
    
    PHt = P_inf @ H.T
    try:
        S_inv = np.linalg.inv(S_inf)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_inf)
    
    K = PHt @ S_inv  # (dim, n_obs)
    
    # Apply Localization to Gain
    K_loc = K.copy()
    for j, obs_idx in enumerate(observed_indices):
        w = _get_loc_weights(obs_idx, dim, _loc_range)
        K_loc[:, j] *= w
    
    # --- 5. Update Mean ---
    mean_update = K_loc @ innovation
    mean_a = mean + mean_update
    
    # --- 6. Update Anomalies (Deterministic EnKF) ---
    # x_i^a = mean_a + (I - K H) (x_i^f - mean^f)
    # Note: We use the *forecast* anomalies here, not the inflated ones, 
    # but we will re-inflate the result to maintain the spread.
    # Operator: (I - K_loc H)
    # (I - K_loc H) is (dim, dim). 
    # anomalies is (N, dim).
    # We need (I - K_loc H) @ anomalies.T -> (dim, N) -> T -> (N, dim)
    
    # Efficiently compute (I - K_loc H) @ anomalies.T
    # K_loc H is (dim, n_obs) @ (n_obs, dim) = (dim, dim)
    KH = K_loc @ H
    # anomalies_T is (dim, N)
    anomalies_a_T = anomalies.T - KH @ anomalies.T
    anomalies_a = anomalies_a_T.T
    
    # --- 7. Post-Inflation ---
    # Scale the updated anomalies by the inflation factor to ensure the ensemble spread
    # reflects the uncertainty.
    anomalies_a = _inflation * anomalies_a
    
    # --- 8. Final Ensemble ---
    updated = mean_a + anomalies_a
    
    # Safety: time guard
    if time.time() - t0 > 100:
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
