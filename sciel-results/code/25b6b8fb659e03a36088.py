"""Ensemble Square Root Filter (EnSRF) for Lorenz-96 data assimilation."""
import numpy as np
import time

def _rk4_step(state, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(state)
    
    def f(x):
        dx = np.empty_like(x)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        return dx
    
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Square Root Filter (EnSRF) analysis update.
    
    Uses sequential coordinate-wise regression (regression-based EnSRF)
    to avoid explicit Kalman gain computation and handle small ensembles.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (16, 10) or (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=np.int64)
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Handle observations shape: could be (16, 10) or (10,)
    if observations.ndim == 1:
        obs_mean = observations
    else:
        obs_mean = observations.mean(axis=0)
    
    # Compute ensemble mean and anomalies
    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean
    
    # Observation noise variance
    R = observation_variance
    
    # Regression-based EnSRF: process each unobserved variable sequentially
    # by regressing it onto the observed variables using ensemble covariance
    
    # Identify unobserved indices
    all_indices = np.arange(n_state)
    unobserved_indices = np.setdiff1d(all_indices, observed_indices)
    
    # Build the observed anomalies matrix (n_members x n_obs)
    obs_anoms = anomalies[:, observed_indices]
    
    # Compute the regression coefficients for each unobserved variable
    # For each unobserved variable j, we want to regress anomaly[j] onto obs_anoms
    # Using the formula: x_j^a = x_j^f + C_jo * C_oo^{-1} * (y - x_o^f)
    # In ensemble form, we can do this via sequential updates
    
    # For numerical stability with small ensembles, we use a regularized approach
    # Compute C_oo (observed-observed covariance)
    C_oo = (obs_anoms.T @ obs_anoms) / (n_members - 1)
    
    # Regularize C_oo for inversion
    reg = 1e-6 * np.trace(C_oo) / n_obs
    C_oo_reg = C_oo + reg * np.eye(n_obs)
    
    # Compute C_oo^{-1}
    try:
        C_oo_inv = np.linalg.inv(C_oo_reg)
    except np.linalg.LinAlgError:
        C_oo_inv = np.linalg.pinv(C_oo_reg)
    
    # Innovation: observation minus forecast mean
    innovation = obs_mean - ens_mean[observed_indices]
    
    # Update observed variables directly
    updated_ensemble = ensemble.copy()
    
    # For observed variables, apply a weighted update
    # Kalman gain for observed variables: K_o = C_oo * (C_oo + R*I)^{-1}
    # In ensemble form, we can use the regression approach
    
    # Sequential EnSRF: process each observed variable one at a time
    # This avoids matrix inversion issues with small ensembles
    
    # Start with the forecast ensemble
    x = ensemble.copy()
    x_mean = x.mean(axis=0)
    
    for obs_idx in observed_indices:
        # Current mean at this observation point
        x_mean_obs = x[:, obs_idx].mean()
        
        # Anomalies for this variable and remaining observations
        x_anom = x[:, obs_idx] - x_mean_obs
        
        # Compute variance of this variable
        var_x = np.var(x_anom, ddof=1)
        
        # Compute covariance with other observed variables (for localization)
        # For simplicity, use a simple inflation/nudging approach for observed vars
        # The key is to update unobserved variables via regression
        
        # Update this observed variable using a simple optimal blending
        # Weight = var_x / (var_x + R)
        weight = var_x / (var_x + R)
        
        # Update the ensemble members
        x[:, obs_idx] = x[:, obs_idx] + weight * (obs_mean - x_mean_obs)
        
        # Now update unobserved variables via regression onto this observed variable
        # For each unobserved variable j:
        #   x_j^a = x_j^f + (C_jo / C_oo) * (y - x_o^f)
        # In ensemble form: x_j += (cov(x_j, x_o) / var(x_o)) * (x_o - x_o_mean)
        
        for j in unobserved_indices:
            x_anom_j = x[:, j] - x[:, j].mean()
            cov_jo = np.dot(x_anom_j, x_anom) / (n_members - 1)
            var_o = np.var(x_anom, ddof=1)
            
            if var_o > 1e-12:
                gain = cov_jo / (var_o + R)
                x[:, j] = x[:, j] + gain * (x[:, obs_idx] - x_mean_obs)
    
    # Check deadline
    if time.time() > deadline:
        return np.clip(x, -100.0, 100.0)
    
    # Apply a light inflation to prevent filter collapse
    x_mean_final = x.mean(axis=0)
    x_anoms_final = x - x_mean_final
    inflation_factor = 1.001
    x = x_mean_final + inflation_factor * x_anoms_final
    
    # Ensure finiteness and bounds
    x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
    x = np.clip(x, -100.0, 100.0)
    
    return x
