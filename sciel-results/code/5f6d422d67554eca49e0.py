"""Lorenz-96 assimilation using an Ensemble Square Root Filter (EnSRF) with sequential coordinate-wise regression."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_step(x, F, dt=0.01):
    """
    One RK4 step for the Lorenz-96 system.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    with periodic boundary conditions.
    """
    n = len(x)
    
    def f(x):
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        return dx
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter (EnSRF) analysis update using sequential
    coordinate-wise regression of unobserved variables onto observed ones.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Compute ensemble anomalies
    anomalies = ens - ens_mean
    
    # Observation error variance
    r_var = observation_variance
    
    # Process each observation sequentially
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        obs_val = observations[j]
        
        # Get the observed component from the ensemble
        ens_obs = ens[:, obs_idx]  # (n_members,)
        ens_obs_mean = np.mean(ens_obs)
        
        # Innovation
        innovation = obs_val - ens_obs_mean
        
        # Compute the ensemble variance of the observed component
        obs_anoms = ens_obs - ens_obs_mean
        obs_var = np.sum(obs_anoms ** 2) / (n_members - 1)
        
        # Innovation variance
        S = obs_var + r_var
        
        # If variance is too small, skip
        if S < 1e-10:
            continue
        
        # Compute the regression coefficients for all state components onto the observed component
        # For each state component i, we want to regress x_i on x_obs
        # The regression coefficient is cov(x_i, x_obs) / var(x_obs)
        
        for i in range(n_state):
            # Compute covariance between state i and observed component
            state_anoms = ens[:, i] - np.mean(ens[:, i])
            
            if i == obs_idx:
                # For the observed component itself, the regression coefficient is 1
                beta = 1.0
            else:
                # Compute covariance
                cov_oi = np.sum(state_anoms * obs_anoms) / (n_members - 1)
                if obs_var > 1e-10:
                    beta = cov_oi / obs_var
                else:
                    beta = 0.0
            
            # Apply localization to the regression coefficient
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            decay_length = 2.5
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            beta_localized = beta * w
            
            # Update the ensemble member values
            # The EnSRF update: x_i^a = x_i^f + beta * (x_obs - x_obs_mean)
            # But we need to account for the fact that we're doing sequential updates
            # and the observed component should be updated to match the observation
            # with some smoothing based on the innovation variance
            
            if i == obs_idx:
                # For the observed component, use the full Kalman gain
                # K_obs = obs_var / (obs_var + r_var)
                K_obs = obs_var / S
                # Update the observed component
                ens[:, i] = ens[:, i] + K_obs * obs_anoms + (1.0 - K_obs) * (innovation * K_obs)
                # Actually, let me think about this more carefully.
                # The standard EnSRF update for the observed variable:
                # x_obs^a = x_obs^f + K * (y - x_obs_mean)
                # where K = P / (P + R)
                # But we want to preserve the ensemble spread.
                # Let's use the standard approach:
                # x_obs^a = x_obs_mean + (1 - K) * (x_obs^f - x_obs_mean) + K * y
                # This preserves the mean and adjusts the spread.
                ens[:, i] = ens_obs_mean + (1.0 - K_obs) * obs_anoms + K_obs * (obs_val - ens_obs_mean)
            else:
                # For unobserved components, use regression
                ens[:, i] = ens[:, i] + beta_localized * obs_anoms
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    
    # Compute the ensemble variance for observed components
    inflation_factors = np.ones(n_state)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        ens_obs_new = ens[:, obs_idx]
        ens_obs_mean_new = np.mean(ens_obs_new)
        obs_anoms_new = ens_obs_new - ens_obs_mean_new
        obs_var_new = np.sum(obs_anoms_new ** 2) / (n_members - 1)
        
        # Expected innovation variance after update should be close to r_var
        # If the ensemble spread is too small, inflate
        if obs_var_new > 1e-10:
            # Target: the ensemble variance should be at least comparable to r_var
            # but not too large
            target_var = r_var * 0.5
            if obs_var_new < target_var:
                ratio = target_var / obs_var_new
                ratio = np.clip(ratio, 1.0, 2.0)
                inflation_factor = np.sqrt(ratio)
            else:
                inflation_factor = 1.0
            
            # Apply to observed component and nearby unobserved components
            for i in range(n_state):
                d = abs(i - obs_idx)
                d = min(d, n_state - d)
                w = np.exp(-d ** 2 / (2.0 * 2.0 ** 2))
                inflation_factors[i] *= (1.0 + 0.5 * w * (inflation_factor - 1.0))
    
    # Global inflation based on overall spread
    # Compute the mean ensemble variance
    all_vars = np.sum(anomalies_new ** 2, axis=0) / (n_members - 1)
    mean_var = np.mean(all_vars)
    
    # A rough estimate of the expected variance from the dynamics
    # For Lorenz-96 with F=8, the typical variance is around 10-20
    # We'll use a conservative inflation
    if mean_var < 5.0:
        global_inflation = np.sqrt(5.0 / max(mean_var, 1e-10))
        global_inflation = np.clip(global_inflation, 1.0, 1.5)
    else:
        global_inflation = 1.0
    
    # Combine component-wise and global inflation
    combined_inflation = 0.7 * inflation_factors + 0.3 * global_inflation
    combined_inflation = np.clip(combined_inflation, 0.95, 1.4)
    
    # Apply inflation around the new mean
    ens = ens_mean_new + combined_inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
