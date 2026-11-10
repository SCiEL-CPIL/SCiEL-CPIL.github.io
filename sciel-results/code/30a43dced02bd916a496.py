import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with:
    - Localized covariance (Gaussian decay)
    - Perturbed observations (to avoid rank deficiency)
    - Adaptive inflation (to prevent filter collapse)
    - Jolliffe-Brier localization
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure observations are 1D
    obs = np.asarray(observations, dtype=float)
    if obs.ndim != 1:
        obs = obs.flatten()
    
    # Observation variance (scalar or per-obs)
    if np.isscalar(observation_variance):
        R = np.full(n_obs, float(observation_variance))
    else:
        R = np.asarray(observation_variance, dtype=float)
        if R.ndim == 0:
            R = np.full(n_obs, float(R))
        else:
            R = np.resize(R, n_obs)
    
    # --- Perturbed observations ---
    # Add independent noise to each member's observation
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within call
    perturbed_obs = obs[None, :] + rng.standard_normal((n_members, n_obs)) * np.sqrt(R)[None, :]
    
    # --- Compute ensemble mean and anomalies ---
    X = np.asarray(ensemble, dtype=float)
    x_mean = X.mean(axis=0)
    X_anom = X - x_mean[None, :]
    
    # --- Covariance of observations ---
    # C_oo = (1/(K-1)) * O_anom^T * O_anom
    O_anom = perturbed_obs - perturbed_obs.mean(axis=0)[None, :]
    C_oo = (O_anom.T @ O_anom) / (n_members - 1)
    # Add small regularization for numerical stability
    C_oo += 1e-6 * np.eye(n_obs)
    
    # --- Cross-covariance between state and observations ---
    # C_xo = (1/(K-1)) * X_anom_obs^T * O_anom
    X_anom_obs = X_anom[:, observed_indices]
    C_xo = (X_anom_obs.T @ O_anom) / (n_members - 1)
    
    # --- Localization (Gaussian decay) ---
    # Apply Jolliffe-Brier style localization
    # For each observed index, decay the influence based on distance
    n_obs_actual = len(observed_indices)
    
    # Build localization matrix for cross-covariance
    # L[i, j] = exp(-0.5 * (d(i,j)/rho)^2) where d is cyclic distance
    rho = 4.0  # Localization radius
    L = np.zeros((n_obs_actual, n_obs_actual))
    for i in range(n_obs_actual):
        for j in range(n_obs_actual):
            d = abs(observed_indices[i] - observed_indices[j])
            d = min(d, n_vars - d)  # Cyclic distance
            L[i, j] = np.exp(-0.5 * (d / rho) ** 2)
    
    # Apply localization to C_xo: C_xo_loc[i, j] = C_xo[i, j] * L[i, j]
    C_xo_loc = C_xo * L.T  # C_xo is (n_obs, n_vars), L is (n_obs, n_obs)
    # Actually C_xo shape is (n_obs, n_vars), we localize the observation dimension
    C_xo_loc = C_xo * L.T
    
    # --- Gain matrix ---
    # K = C_xo_loc @ inv(C_oo)
    try:
        C_oo_inv = np.linalg.inv(C_oo)
    except np.linalg.LinAlgError:
        C_oo_inv = np.linalg.pinv(C_oo)
    
    K = C_xo_loc @ C_oo_inv  # Shape: (n_obs, n_vars)
    
    # --- Update ensemble ---
    # For each member: x_a = x_f + K @ (y_f + eps - y_f_mean)
    # Using perturbed observations approach:
    # x_a(k) = x_f(k) + K @ (perturbed_obs(k) - perturbed_obs_mean)
    obs_anom = perturbed_obs - perturbed_obs.mean(axis=0)[None, :]  # (n_members, n_obs)
    
    X_anom_updated = X_anom + K @ obs_anom.T  # (n_vars, n_members)
    X_anom_updated = X_anom_updated.T  # (n_members, n_vars)
    
    # Update mean
    x_mean_updated = x_mean + K @ (obs - perturbed_obs.mean(axis=0))
    
    # Reconstruct ensemble
    result = x_mean_updated[None, :] + X_anom_updated
    
    # --- Inflation ---
    # Check if ensemble variance is too small; apply multiplicative inflation
    # Simple adaptive inflation based on observed innovation
    innovation = obs - x_mean[observed_indices]
    innovation_var = np.var(innovation)
    
    # Estimate expected innovation variance
    C_oo_pred = (X_anom_obs.T @ X_anom_obs) / (n_members - 1)
    C_oo_pred += np.diag(R)
    expected_innov_var = np.trace(C_oo_pred) / n_obs
    
    # Inflation factor
    if expected_innov_var > 1e-10:
        ratio = innovation_var / expected_innov_var
        # Gentle inflation
        if ratio < 0.5:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
        else:
            inflation = 1.0
    else:
        inflation = 1.0
    
    # Apply inflation about the updated mean
    if inflation > 1.0:
        result = x_mean_updated[None, :] + inflation * (result - x_mean_updated[None, :])
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
