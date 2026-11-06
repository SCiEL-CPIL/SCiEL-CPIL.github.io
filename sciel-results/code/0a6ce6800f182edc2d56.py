"""Ensemble Kalman filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # --- Compute ensemble statistics ---
    ens_mean = np.mean(ensemble, axis=0)  # (20,)
    
    # Anomalies
    anomalies = ensemble - ens_mean  # (16, 20)
    
    # Ensemble covariance (sample covariance with (nm-1) normalization)
    # C = anomalies^T @ anomalies / (nm - 1)
    C = (anomalies.T @ anomalies) / (nm - 1)  # (20, 20)
    
    # --- Localization: taper covariance by distance ---
    # For Lorenz-96 with periodic boundary, distance is min(|i-j|, nvars-|i-j|)
    loc_radius = 4  # localization radius
    loc_width = 2.5  # width of taper
    
    dist_matrix = np.zeros((nvars, nvars))
    for i in range(nvars):
        for j in range(nvars):
            d = abs(i - j)
            d = min(d, nvars - d)
            dist_matrix[i, j] = d
    
    # Gaspari-Cohn-like taper
    loc_weights = np.zeros((nvars, nvars))
    for i in range(nvars):
        for j in range(nvars):
            d = dist_matrix[i, j]
            if d <= loc_radius:
                loc_weights[i, j] = 1.0
            elif d <= loc_radius + loc_width:
                # Smooth taper
                t = (d - loc_radius) / loc_width
                loc_weights[i, j] = 0.5 * (1.0 + np.cos(np.pi * t))
            else:
                loc_weights[i, j] = 0.0
    
    # Apply localization to covariance
    C_loc = C * loc_weights
    
    # --- Observation operator ---
    # H maps state to observation space: H x = x[observed_indices]
    H = np.zeros((nobs, nvars))
    for k, idx in enumerate(observed_indices):
        H[k, int(idx)] = 1.0
    
    # --- Compute Kalman gain ---
    # P_oo = H C_loc H^T + R
    HCHt = H @ C_loc @ H.T  # (10, 10)
    R = obs_var * np.eye(nobs)
    P_oo = HCHt + R  # (10, 10)
    
    # Innovation
    innovation = observations - H @ ens_mean  # (10,)
    
    # Invert P_oo
    try:
        P_oo_inv = np.linalg.inv(P_oo)
    except np.linalg.LinAlgError:
        P_oo_inv = np.linalg.pinv(P_oo)
    
    # Kalman gain: K = C_loc H^T P_oo^{-1}
    K = (C_loc @ H.T) @ P_oo_inv  # (20, 10)
    
    # --- Adaptive inflation ---
    # Check if the ensemble spread is consistent with the innovations
    # If spread is too small, inflate
    P_oo_diag = np.diag(P_oo)
    innovation_var = np.var(innovation, ddof=1) if nm > 1 else 1.0
    # Expected innovation variance from ensemble
    expected_innov_var = np.mean(P_oo_diag)
    
    # Inflation factor: if spread < innovation variance, inflate
    if expected_innov_var > 1e-10:
        ratio = innovation_var / expected_innov_var
        # Gentle inflation
        inflation = 1.0 + 0.1 * max(0.0, ratio - 1.0)
        inflation = min(inflation, 1.3)  # cap inflation
    else:
        inflation = 1.0
    
    # Apply inflation to anomalies
    anomalies_inflated = anomalies * np.sqrt(inflation)
    
    # --- Update ensemble members ---
    # For each member: x_a = x_f + K @ (y - H x_f)
    # Using the inflated anomalies for the perturbation part
    result = np.zeros_like(ensemble)
    for m in range(nm):
        x_f = ens_mean + anomalies_inflated[m]  # (20,)
        Hx_f = H @ x_f  # (10,)
        innov_m = observations - Hx_f  # (10,)
        x_a = x_f + K @ innov_m  # (20,)
        result[m] = x_a
    
    # --- Safety: clamp to [-100, 100] and ensure finiteness ---
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    if time.time() - t0 > 110.0:
        pass  # We're already done, just return
    
    return result
# EVOLVE-BLOCK-END
