import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Kalman Filter update with localization and adaptive inflation.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array.
    """
    # Time guard
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin for 120s limit

    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)

    # 1. Compute Ensemble Mean and Anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # 2. Compute Ensemble Covariance at observed locations
    # C_oo = (1/(N-1)) * A_o^T A_o
    A_obs = anomalies[:, obs_idx]  # (N, n_obs)
    C_oo = (A_obs.T @ A_obs) / (N - 1)
    
    # Add observation noise to denominator matrix
    R = observation_variance * np.eye(n_obs)
    S = C_oo + R  # Innovation covariance
    
    # 3. Compute Kalman Gain
    # K = C_xo * S^{-1}
    # C_xo = (1/(N-1)) * A^T A_o
    A_all = anomalies  # (N, D)
    C_xo = (A_all.T @ A_obs) / (N - 1)  # (D, n_obs)
    
    # Solve K * S = C_xo using Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        # Solve L * Y^T = C_xo^T => Y^T = L^{-1} C_xo^T
        # Then K = Y * L^{-T}
        Y = np.linalg.solve(L, C_xo.T).T  # (n_obs, D)
        K = Y @ np.linalg.inv(L.T)  # (n_obs, D) -> Actually K should be (D, n_obs)
        # Let's stick to standard matrix inversion for small n_obs (10) to avoid confusion in transposes
        # S is 10x10, inversion is cheap.
        S_inv = np.linalg.inv(S)
        K = C_xo @ S_inv  # (D, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        S_inv = np.linalg.pinv(S)
        K = C_xo @ S_inv

    # 4. Localization
    # Apply exponential localization to the gain to prevent spurious correlations
    # Distance in cyclic space
    loc_dist = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)  # Cyclic distance
            loc_dist[i, j] = dist
    
    # Localization length scale (tuned for L96, typically ~4-6)
    L_loc = 5.0
    loc_weights = np.exp(-loc_dist / L_loc)
    
    # Apply localization to K
    K_localized = K * loc_weights  # (D, n_obs)

    # 5. Adaptive Inflation
    # Estimate if the ensemble is under-dispersive
    # Compare innovation norm to expected norm
    innovation = observations - mean[obs_idx]  # (n_obs,)
    
    # Expected innovation variance is trace(S)
    expected_var = np.trace(S)
    actual_var = np.dot(innovation, innovation)
    
    # If actual_var > expected_var, inflation is needed
    # Simple adaptive inflation: multiply anomalies by alpha
    # alpha = sqrt(actual_var / expected_var) capped to avoid blowup
    if expected_var > 1e-6:
        ratio = actual_var / expected_var
        # Only inflate if significantly larger, dampen if smaller? 
        # Standard approach: inflate if ratio > 1.
        alpha = np.sqrt(ratio)
        # Cap inflation to prevent instability
        alpha = np.clip(alpha, 1.0, 2.0)
        
        # Inflate anomalies
        anomalies_inflated = anomalies * alpha
        # Recalculate mean? Mean stays same if we just scale anomalies around mean.
        # But usually we inflate the whole ensemble around the mean.
        # Let's apply inflation to the anomalies before updating.
        # Wait, the standard EnKF update is:
        # x_new = x_old + K (y - H x_old)
        # This assumes the prior is correct. If we inflate, we should use the inflated ensemble for the gain calculation?
        # Strictly, inflation is applied to the prior ensemble before the update.
        # Let's redo the calculation with inflated anomalies if alpha > 1.0
        if alpha > 1.0:
            A_obs_inf = anomalies_inflated[:, obs_idx]
            C_oo_inf = (A_obs_inf.T @ A_obs_inf) / (N - 1)
            S_inf = C_oo_inf + R
            try:
                S_inv_inf = np.linalg.inv(S_inf)
            except:
                S_inv_inf = np.linalg.pinv(S_inf)
            
            A_all_inf = anomalies_inflated
            C_xo_inf = (A_all_inf.T @ A_obs_inf) / (N - 1)
            K_inf = C_xo_inf @ S_inv_inf
            K_localized = K_inf * loc_weights
    else:
        K_localized = K * loc_weights

    # 6. Update Ensemble
    # Innovation
    innov = observations - mean[obs_idx]  # (n_obs,)
    
    # Update mean
    mean_update = K_localized @ innov  # (D,)
    new_mean = mean + mean_update
    
    # Update anomalies (Ensemble Transform or simple linear update)
    # Simple linear update: x_i' = x_i + K (y - H x_i)
    # This is the "Local" EnKF approach, which is robust.
    # H x_i is just x_i[obs_idx]
    # x_i[obs_idx] is (n_obs,)
    # K is (D, n_obs)
    # So correction_i = K @ (y - x_i[obs_idx])
    
    updated_ensemble = ensemble.copy()
    for i in range(N):
        x_i_obs = ensemble[i, obs_idx]
        corr = K_localized @ (observations - x_i_obs)
        updated_ensemble[i] += corr

    # 7. Clip to bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check time
    if time.time() > deadline:
         return updated_ensemble

    return updated_ensemble
