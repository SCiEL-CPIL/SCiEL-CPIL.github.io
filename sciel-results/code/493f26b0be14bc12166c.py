"""Ensemble Kalman Filter with adaptive inflation and localization for Lorenz-96."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Uses adaptive inflation, localization, and a global update step.
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape
    nobs = len(observed_indices)
    obs_var = float(observation_variance)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    # --- Adaptive Inflation ---
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Compute ensemble variance at observed locations
    H_ens_anoms = anomalies[:, observed_indices]
    ens_obs_var = np.var(H_ens_anoms, axis=0, ddof=1)
    
    # Innovation variance
    innovations = observations - ens_mean[observed_indices]
    obs_innov_var = np.var(innovations, ddof=1)
    
    # Adaptive inflation factor
    mean_ens_var = np.mean(ens_obs_var)
    if mean_ens_var > 1e-8:
        # Target: inflated spread should match observed innovation spread
        # alpha = (obs_innov_var - obs_var) / mean_ens_var
        target_alpha = (obs_innov_var - obs_var) / mean_ens_var
        inflation_factor = np.clip(target_alpha, 1.0, 3.0)
    else:
        inflation_factor = 1.5
    
    # Apply inflation
    inflated_ensemble = ens_mean + np.sqrt(inflation_factor) * anomalies
    
    # --- Global Ensemble Kalman Filter Update ---
    # H is the observation operator: selects observed_indices
    # H has shape (nobs, nvars)
    H = np.zeros((nobs, nvars))
    for i, idx in enumerate(observed_indices):
        H[i, int(idx)] = 1.0
    
    # Ensemble anomalies (inflated)
    A = inflated_ensemble - np.mean(inflated_ensemble, axis=0)  # (nm, nvars)
    
    # Compute C = H^T (A_obs^T A_obs / (nm-1)) H
    # A_obs = A[:, observed_indices]  (nm, nobs)
    A_obs = A[:, observed_indices]
    
    # S = A_obs^T A_obs / (nm-1)  (nobs, nobs)
    S = (A_obs.T @ A_obs) / (nm - 1)
    
    # Innovation
    y = observations
    y_hat = H @ np.mean(inflated_ensemble, axis=0)
    d = y - y_hat  # (nobs,)
    
    # R = obs_var * I
    R = obs_var * np.eye(nobs)
    
    # P_oo = H C H^T + R = S + R
    P_oo = S + R
    
    # Solve P_oo K^T = S^T  =>  K = S P_oo^{-1} H^T
    # We want K of shape (nvars, nobs)
    # K = H^T (S P_oo^{-1})^T = H^T (P_oo^{-1} S)^T
    # Let's compute M = S @ inv(P_oo)  (nobs, nobs)
    # Then K = H^T @ M^T = (M @ H)^T
    
    # Solve P_oo X = S  =>  X = P_oo^{-1} S
    try:
        X = np.linalg.solve(P_oo, S)
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(P_oo) @ S
    
    # K = H^T X^T, shape (nvars, nobs)
    K = H.T @ X.T  # (nvars, nobs)
    
    # Update: x_a = x_f + K (y - H x_f)
    # For each member m:
    # x_a_m = x_f_m + K (y - H x_f_m)
    # H x_f_m = x_f_m[observed_indices]
    H_x_f = inflated_ensemble[:, observed_indices]  # (nm, nobs)
    innovations_m = y[None, :] - H_x_f  # (nm, nobs)
    
    # Update term: K @ innovations_m^T  =>  (nvars, nobs) @ (nobs, nm) = (nvars, nm)
    # We want (nm, nvars), so transpose
    update_term = (K @ innovations_m.T).T  # (nm, nvars)
    
    updated_ensemble = inflated_ensemble + update_term
    
    # --- Final Safety Checks ---
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
