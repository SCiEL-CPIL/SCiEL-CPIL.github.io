"""Lorenz-96 ensemble assimilation using Localized EnKF with Adaptive Inflation."""
import numpy as np
import time

def _build_localization_matrix(N, L):
    """Build Gaspari-Cohn localization matrix for a ring of N variables with correlation length L."""
    def gc(d, L):
        """Gaspari-Cohn 5th-order correlation function."""
        d = np.abs(d)
        r = d / L
        result = np.zeros_like(d)
        
        # Piecewise polynomial
        # r in [0, 0.5]
        mask1 = r <= 0.5
        r1 = r[mask1]
        result[mask1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
        
        # r in (0.5, 1.5]
        mask2 = (r > 0.5) & (r <= 1.5)
        r2 = r[mask2]
        result[mask2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
        
        # r in (1.5, 2.5]
        mask3 = (r > 1.5) & (r <= 2.5)
        r3 = r[mask3]
        result[mask3] = (-1.0/12.0)*r3**5 + (5.0/4.0)*r3**4 - (5.0/2.0)*r3**3 + 5.0*r3**2 - (25.0/12.0)*r3 + 35.0/12.0
        
        # r > 2.5 is 0
        
        return result
    
    loc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Ring distance
            d = min(abs(i - j), N - abs(i - j))
            loc[i, j] = gc(d, L)
    return loc

# Precompute localization matrix with L=2.0 (tighter than previous L=3.0)
_N = 20
_L = 2.0
_LOC = _build_localization_matrix(_N, _L)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter (EnKF) with Adaptive Inflation.
    
    Parameters:
    -----------
    ensemble : (16, 20) array
        Ensemble of state vectors.
    observations : (16, 10) array
        Noisy observations of even-indexed coordinates.
    observed_indices : array of indices
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : (16, 20) array
        Updated ensemble.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    ens = ensemble.copy().astype(float)
    n_ens, n_state = ens.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Step 1: Compute ensemble mean and anomalies
    ens_mean = ens.mean(axis=0)
    anomalies = ens - ens_mean  # (16, 20)
    
    # Step 2: Compute Localized Background Covariance
    # We need P_bg_loc for the observed variables and their cross-covariance with all variables?
    # Standard EnKF update:
    # x_a = x_b + K (y - H x_b)
    # For ensemble:
    # x_a_i = x_b_i + K (y_i - H x_b_i)
    # The gain K is computed from the localized covariance.
    
    # Extract anomalies at observed locations
    X_obs = anomalies[:, obs_idx]  # (16, 10)
    
    # Compute background covariance at observed locations
    P_obs_obs = (X_obs.T @ X_obs) / (n_ens - 1)  # (10, 10)
    
    # Apply localization to P_obs_obs
    P_obs_obs_loc = P_obs_obs.copy()
    for i in range(n_obs):
        for j in range(n_obs):
            P_obs_obs_loc[i, j] *= _LOC[obs_idx[i], obs_idx[j]]
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance
    S = P_obs_obs_loc + R  # (10, 10)
    
    # Compute Kalman Gain
    # K = P_obs_obs_loc @ S^{-1}
    # We need the full gain matrix K_full of shape (20, 10) to update all variables.
    # K_full[:, j] = sum_i P_loc[i, obs_idx[j]] S^{-1}_{j, i}
    # P_loc[i, k] = P_bg[i, k] * LOC[i, k]
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Compute cross-covariance between all state variables and observed variables, localized
    # P_cross[i, j_obs] = (1/(n-1)) * sum_m anomalies[m, i] * anomalies[m, j_obs] * LOC[i, j_obs]
    # anomalies is (16, 20), X_obs is (16, 10)
    # P_cross_raw = (anomalies.T @ X_obs) / (n_ens - 1)  # (20, 10)
    P_cross_raw = (anomalies.T @ X_obs) / (n_ens - 1)
    
    P_cross_loc = P_cross_raw.copy()
    for i in range(n_state):
        for j in range(n_obs):
            P_cross_loc[i, j] *= _LOC[i, obs_idx[j]]
    
    # Full Kalman Gain K (20, 10)
    K = P_cross_loc @ S_inv
    
    # Step 3: Update Ensemble Mean
    # Innovation for the mean
    innovation_mean = observations.mean(axis=0) - ens_mean[obs_idx]  # (10,)
    # Note: observations is (16, 10). Mean over ensemble members.
    
    ens_mean_new = ens_mean.copy()
    ens_mean_new += K @ innovation_mean
    
    # Step 4: Update Anomalies
    # The update for each member's anomaly is:
    # anomaly_new = anomaly - K @ (H anomaly)
    # H anomaly is just the anomaly at observed indices: X_obs (16, 10)
    # K is (20, 10)
    # update = K @ X_obs.T  # (20, 16)
    update = K @ X_obs.T
    anomalies_new = anomalies - update.T  # (16, 20)
    
    # Step 5: Adaptive Inflation
    # We want the spread of the updated ensemble at observed points to match the spread of the observations.
    # Since observations are independent N(0, 0.49), their variance is 0.49.
    # However, the "spread" of the ensemble should reflect the posterior uncertainty.
    # A common adaptive inflation method:
    # Calculate the standard deviation of the updated ensemble at observed points.
    # Calculate the standard deviation of the innovations (y - H x_b).
    # Or simply match the ensemble variance to the observation variance if the system is well-calibrated.
    # Better: Match the ensemble spread to the innovation spread.
    
    # Innovation for each member: y_m - H x_b_m
    # y_m is observations[m, :] (10,)
    # H x_b_m is ens[m, obs_idx] (10,)
    innovations_ens = observations - ens[:, obs_idx]  # (16, 10)
    innovation_var = np.var(innovations_ens, axis=0)  # (10,)
    
    # Updated ensemble spread at observed points
    updated_ens_obs = ens_mean_new[obs_idx] + anomalies_new[:, obs_idx]
    updated_var = np.var(updated_ens_obs, axis=0)  # (10,)
    
    # Inflation factor alpha
    # We want var(alpha * anomalies + mean) ~ var(innovations) ?
    # Actually, we want the posterior variance to be consistent.
    # Simple adaptive inflation:
    # alpha = sqrt(mean(innovation_var) / mean(updated_var))
    # But we must be careful not to inflate too much.
    
    mean_innov_var = np.mean(innovation_var)
    mean_updated_var = np.mean(updated_var)
    
    if mean_updated_var > 1e-6:
        alpha = np.sqrt(mean_innov_var / mean_updated_var)
    else:
        alpha = 1.0
    
    # Clip inflation to reasonable bounds to prevent divergence or collapse
    # Typically 1.0 to 1.2
    alpha = np.clip(alpha, 1.0, 1.2)
    
    # Apply inflation to anomalies
    # We also need to ensure the mean doesn't drift. Inflation is usually applied to anomalies around the mean.
    ens_final = ens_mean_new + alpha * anomalies_new
    
    # Ensure finite values
    ens_final = np.nan_to_num(ens_final, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to valid range
    ens_final = np.clip(ens_final, -100.0, 100.0)
    
    return ens_final
