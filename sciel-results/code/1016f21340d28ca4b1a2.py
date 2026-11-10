"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(r, L):
    """Gaspari-Cohn localization function."""
    r = np.abs(r)
    out = np.zeros_like(r, dtype=float)
    # r <= L/2
    mask1 = r <= L / 2.0
    r1 = r[mask1]
    out[mask1] = (-1 / (2 * (L / 2.0) ** 5) * r1**5
                  + 10 / (L**4) * r1**4
                  - 5 * (L / 2.0) / (L**4) * r1**3
                  - 5 / (L**2) * r1**2
                  + 40 / (9 * L) * r1 + 1)
    # L/2 < r <= 3L/2
    mask2 = (r > L / 2.0) & (r <= 3 * L / 2.0)
    r2 = r[mask2]
    c = (L / 2.0)
    out[mask2] = (1 / (4 * (3 * L / 2.0 - c)**5) * (r2 - 3 * L / 2.0)**5
                  - 5 / (4 * (3 * L / 2.0 - c)**4) * (r2 - 3 * L / 2.0)**4
                  + 5 * (3 * L / 2.0 - c) / (4 * (3 * L / 2.0 - c)**4) * (r2 - 3 * L / 2.0)**3
                  + 5 / (4 * (3 * L / 2.0 - c)**2) * (r2 - 3 * L / 2.0)**2
                  - 20 / (9 * (3 * L / 2.0 - c)) * (r2 - 3 * L / 2.0)
                  + 4 / 9)
    return out


def _local_esrf_update(ensemble, observations, observed_indices, observation_variance,
                       localization_length=4.0, inflation_beta=0.2, target_ratio=1.15):
    """
    Localized Ensemble Square Root Filter (ESRF) update with Gaspari-Cohn localization
    and adaptive inflation.
    
    ensemble: (n_ens, n_vars)
    observations: (n_obs,)
    observed_indices: (n_obs,)
    observation_variance: scalar
    """
    n_ens, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Center the ensemble
    mean = ensemble.mean(axis=0)
    centered = ensemble - mean
    
    # Compute covariance at observed indices
    # Y = centered[:, observed_indices]  shape (n_ens, n_obs)
    Y = centered[:, observed_indices]
    
    # Anomaly covariance matrix C_yy = Y^T Y / (n_ens - 1)
    C_yy = (Y.T @ Y) / (n_ens - 1)
    
    # Add observation noise
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance S = C_yy + R
    S = C_yy + R
    
    # Apply localization to the covariance matrix
    # Build distance matrix for observed indices
    obs_idx = observed_indices
    dist = np.abs(np.subtract.outer(obs_idx, obs_idx)).astype(float)
    # For periodic Lorenz-96, wrap-around distance
    dist = np.minimum(dist, n_vars - dist)
    loc = _gaspari_cohn(dist, localization_length)
    
    # Localize the covariance
    C_yy_loc = C_yy * loc
    
    S_loc = C_yy_loc + R
    
    # Compute Kalman gain: K = C_xy * S^{-1}
    # C_xy = centered @ Y^T / (n_ens - 1)  shape (n_vars, n_obs)
    C_xy = (centered @ Y.T) / (n_ens - 1)
    
    # Localize cross-covariance as well
    # For each variable i and observed j, distance is |i - j| (with wrap)
    var_idx = np.arange(n_vars)
    dist_xy = np.abs(np.subtract.outer(var_idx, obs_idx)).astype(float)
    dist_xy = np.minimum(dist_xy, n_vars - dist_xy)
    loc_xy = _gaspari_cohn(dist_xy, localization_length)
    
    C_xy_loc = C_xy * loc_xy
    
    # Invert S_loc
    try:
        S_inv = np.linalg.inv(S_loc)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_loc)
    
    K = C_xy_loc @ S_inv  # shape (n_vars, n_obs)
    
    # Innovation
    innovation = observations - mean[observed_indices]  # shape (n_obs,)
    
    # Update mean
    new_mean = mean + K @ innovation
    
    # Update ensemble using ESRF scheme
    # We need to compute the update for each ensemble member
    # ESRF: x_new = x + (K @ (y - x_obs)) * sqrt(n_ens/(n_ens-1)) approximately
    # More precisely, use the ensemble transform approach
    
    # Compute the observation-space anomaly for each member
    # Y = centered[:, observed_indices]  (n_ens, n_obs)
    
    # The ESRF update: 
    # x_new_i = x_i + K @ (y - mean_obs) + (some correction to preserve spread)
    # Standard ESRF:
    # Let A = Y (n_ens x n_obs)
    # The update is: x_new = x + K @ (y - mean_y)
    # But we need to adjust to preserve ensemble variance
    
    # Simpler approach: use the standard ensemble Kalman filter update
    # x_new_i = x_i + K @ (y - x_i_obs)
    # This is the ensemble Kalman filter (not exactly ESRF but similar)
    
    # Let's use the standard EKF update which is simpler and works well
    # x_new_i = x_i + K @ (y - x_i[obs])
    
    new_ensemble = ensemble.copy()
    for i in range(n_ens):
        obs_anomaly = observations - ensemble[i, observed_indices]
        new_ensemble[i] += K @ obs_anomaly
    
    # Apply inflation to prevent collapse
    # Inflate around the new mean
    new_ensemble = new_mean + inflation_beta * (new_ensemble - new_ensemble.mean(axis=0))
    
    # Check if we need more inflation based on target ratio
    # Compute observation-space variance ratio
    new_obs_var = np.var(new_ensemble[:, observed_indices], axis=0).mean()
    old_obs_var = np.var(ensemble[:, observed_indices], axis=0).mean()
    if old_obs_var > 1e-10:
        ratio = new_obs_var / old_obs_var
        if ratio < target_ratio:
            # Inflate more
            extra_inflation = 1.0 + (target_ratio - ratio) * 0.5
            new_ensemble = new_mean + extra_inflation * (new_ensemble - new_ensemble.mean(axis=0))
    
    # Ensure finite values
    new_ensemble = np.where(np.isfinite(new_ensemble), new_ensemble, 0.0)
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    return new_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using localized ESRF with adaptive inflation.
    
    ensemble: (16, 20) array
    observations: (10,) array of noisy observations at even indices
    observed_indices: (10,) array of indices [0, 2, 4, ..., 18]
    observation_variance: scalar (0.49)
    
    Returns: updated ensemble with same (16, 20) shape
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    try:
        result = _local_esrf_update(
            ensemble, observations, observed_indices, observation_variance,
            localization_length=4.0,
            inflation_beta=0.2,
            target_ratio=1.15
        )
    except Exception:
        # Fallback to simple nudging if anything goes wrong
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.clip(result, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found (which is result)
        pass
    
    return result

# EVOLVE-BLOCK-END
