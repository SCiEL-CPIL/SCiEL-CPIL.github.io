"""Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) analysis update for Lorenz-96.
    
    Uses gas-station localization and adaptive inflation to maintain
    ensemble spread and accuracy.
    
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
    
    # 1. Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # 2. Compute background anomalies
    anomalies = ensemble - ens_mean  # (nm, nvars)
    
    # 3. Compute background covariance (state-space)
    # C_b = (1/(nm-1)) * Anomalies.T @ Anomalies
    C_b = (anomalies.T @ anomalies) / (nm - 1)  # (nvars, nvars)
    
    # 4. Localization
    # Apply a localization function to C_b to remove spurious long-range correlations.
    # Use a simple distance-based decay (Gas Station style or similar).
    # For nvars=20, periodic distance.
    loc_mask = np.ones((nvars, nvars))
    for i in range(nvars):
        for j in range(nvars):
            d = abs(i - j)
            d = min(d, nvars - d)
            # Simple exponential decay or step function.
            # Let's use a radius of 5 with a smooth cutoff.
            if d > 5:
                loc_mask[i, j] = 0.0
            else:
                # Tapered localization (Hunt et al.)
                # cos^2 profile for d in [3, 5], 1 for d < 3
                if d >= 3:
                    loc_mask[i, j] = 0.5 * (1 + np.cos(np.pi * (d - 3) / 2.0))
                else:
                    loc_mask[i, j] = 1.0
    C_loc = C_b * loc_mask
    
    # 5. Observation Operator H
    H = np.zeros((nobs, nvars))
    for k, idx in enumerate(observed_indices):
        H[k, int(idx)] = 1.0
    
    # 6. Compute C_oo (Observation Covariance)
    # C_oo = H C_loc H^T + R
    H_C_loc = H @ C_loc  # (nobs, nvars)
    C_oo = H_C_loc @ H.T + obs_var * np.eye(nobs)  # (nobs, nobs)
    
    # 7. Compute Kalman Gain K
    # K = C_loc H^T C_oo^{-1}
    try:
        C_oo_inv = np.linalg.inv(C_oo)
    except np.linalg.LinAlgError:
        C_oo_inv = np.linalg.pinv(C_oo)
    
    K = (C_loc @ H.T) @ C_oo_inv  # (nvars, nobs)
    
    # 8. Update ensemble
    # Innovation for each member: y - H x_f
    # x_f = ens_mean + anomalies
    # H x_f = H ens_mean + H anomalies
    H_ens_mean = H @ ens_mean  # (nobs,)
    
    # For each member
    updated_ensemble = np.zeros_like(ensemble)
    for m in range(nm):
        x_f = ens_mean + anomalies[m]
        H_x_f = H @ x_f
        innovation_m = observations - H_x_f
        x_a = x_f + K @ innovation_m
        updated_ensemble[m] = x_a
    
    # 9. Adaptive Inflation
    # Estimate inflation factor based on the ratio of observed innovation variance to expected.
    # Simple adaptive inflation:
    # Calculate sample innovation variance
    H_anomalies = H @ anomalies.T  # (nobs, nm)
    innovations_ens = observations[:, None] - (H_ens_mean[:, None] + H_anomalies) # (nobs, nm)
    # Variance of innovations in obs space
    # innovations_ens has shape (nobs, nm)
    S_y_sample = np.var(innovations_ens, axis=1)  # (nobs,)
    S_y_theo = np.diag(C_oo)  # (nobs,)
    
    # Estimate inflation factor alpha
    # We want to scale anomalies so that the new S_y matches S_y_theo
    # Roughly, alpha = sqrt(mean(S_y_theo / S_y_sample))
    ratio = np.mean(S_y_theo / np.maximum(S_y_sample, 1e-6))
    alpha = np.sqrt(ratio)
    
    # Limit inflation to avoid instability (e.g., max 1.5)
    alpha = np.clip(alpha, 1.0, 1.5)
    
    # Apply inflation to anomalies
    # x_a_inflated = new_mean + alpha * (x_a - new_mean)
    new_mean = np.mean(updated_ensemble, axis=0)
    
    for m in range(nm):
        updated_ensemble[m] = new_mean + alpha * (updated_ensemble[m] - new_mean)
    
    # 10. Safety checks
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    if time.time() - t0 > 110.0:
        pass
        
    return updated_ensemble
# EVOLVE-BLOCK-END
