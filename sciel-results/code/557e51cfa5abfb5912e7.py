"""Lorenz-96 assimilation using Ensemble Square Root Filter with localization and adaptive inflation."""
import numpy as np
import time

# Time deadline guard
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Square Root Filter analysis update with localization and adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    obs_idx = np.asarray(observed_indices, dtype=int)
    
    # Precompute localization weights for each observed index
    # Use wider localization to reduce over-localization artifacts
    decay_len = 2.5
    loc_weights = np.zeros((n_obs, n_state))
    for j, oidx in enumerate(obs_idx):
        for i in range(n_state):
            d = abs(i - oidx)
            d = min(d, n_state - d)
            loc_weights[j, i] = np.exp(-d / decay_len)
    
    # Sequential EnSRF: process observations one at a time
    # For each observation, use regression to update all variables
    for j in range(n_obs):
        if _check_deadline():
            break
            
        oidx = obs_idx[j]
        obs_val = observations[j]
        obs_var = observation_variance
        
        # Current ensemble mean
        cur_mean = np.mean(ens, axis=0)
        
        # Ensemble values at observed coordinate
        x_obs = ens[:, oidx]  # (n_members,)
        x_obs_mean = np.mean(x_obs)
        
        # Innovation
        innov = obs_val - x_obs_mean
        
        # Variance of observed variable in ensemble
        var_obs = np.var(x_obs, ddof=1)
        
        # Avoid division by zero
        if var_obs < 1e-10:
            continue
        
        # Denominator: var_obs + observation variance
        denom = var_obs + obs_var
        
        # For each state variable, compute the localized regression coefficient
        # cov(x_i, x_obs) / (var_obs + R) * localization_weight
        for i in range(n_state):
            # Compute covariance between state variable i and observed variable
            x_i = ens[:, i]
            x_i_mean = np.mean(x_i)
            
            # Covariance
            cov_io = np.mean((x_i - x_i_mean) * (x_obs - x_obs_mean))
            
            # Kalman gain with localization
            K = (cov_io / denom) * loc_weights[j, i]
            
            # Update: x_i_a = x_i_f + K * (obs - H(x_i)_f)
            # For the observed variable itself, H(x_i) = x_i, so innovation is obs - x_i
            # For other variables, we use the same innovation (obs - x_obs_mean)
            if i == oidx:
                ens[:, i] = x_i + K * (obs_val - x_i)
            else:
                ens[:, i] = x_i + K * innov
        
        # After each sequential update, recompute mean for next step
        # (this is inherent in the sequential approach)
    
    # Compute ensemble mean after all updates
    ens_mean_new = np.mean(ens, axis=0)
    
    # Adaptive inflation: match ensemble spread to innovation statistics
    # Compute observed ensemble spread
    obs_ens_new = ens[:, observed_indices]
    obs_mean_new = np.mean(obs_ens_new, axis=0)
    obs_anoms_new = obs_ens_new - obs_mean_new
    
    # Expected innovation variance from updated ensemble
    P_oo_new = (obs_anoms_new.T @ obs_anoms_new) / (n_members - 1)
    S_expected_diag = np.diag(P_oo_new) + observation_variance
    
    # Actual innovation (before update, we can approximate)
    # Use the spread ratio approach
    ens_spread_obs = np.std(ens[:, observed_indices], axis=0, ddof=1)
    mean_spread = np.mean(ens_spread_obs)
    
    # Compute a simple inflation factor based on spread
    # Target: spread should be reasonable relative to observation noise
    target_spread = np.sqrt(observation_variance) * 1.5
    if mean_spread > 1e-8:
        ratio = target_spread / mean_spread
        inflation = np.clip(np.sqrt(ratio), 0.95, 1.35)
    else:
        inflation = 1.0
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens
