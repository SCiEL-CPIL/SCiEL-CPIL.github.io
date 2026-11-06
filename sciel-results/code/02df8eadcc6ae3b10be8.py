"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _finite_clip(arr, lo=-100.0, hi=100.0):
    """Ensure array is finite and within bounds."""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, lo, hi)

def _gaspari_cohn(d, b=1.0, c=2.0):
    """Gaspari-Cohn correlation function."""
    d = np.abs(d)
    b = np.abs(b)
    c = np.abs(c)
    # Standard GC function
    with np.errstate(divide='ignore', invalid='ignore'):
        gc = np.where(
            d <= b,
            -0.25 * d**4 / b**4 + 0.5 * d**2 / b**2 + 1.0,
            np.where(
                d <= c,
                (0.25 * (2*c - d)**4 / (b * (c - b)))
                - (2*c - d)**2 / (b * (c - b))
                + 5.0/3.0
                + 4.0/(c - b)
                - 2.0 * (c - b) / (b * c),
                0.0
            )
        )
    return np.maximum(gc, 0.0)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 partial observation assimilation.
    
    Uses ensemble Kalman filter with Gaspari-Cohn localization and adaptive inflation.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    result = ensemble.copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Observation innovation
    obs_innov = observations - ens_mean[observed_indices]
    
    # Compute ensemble covariance for observed variables
    # P_o = (1/(n-1)) * sum_i (x_i - mean)(x_i - mean)^T for observed indices
    anomalies = ensemble - ens_mean
    # Covariance of observed components
    P_oo = np.cov(anomalies[:, observed_indices], rowvar=False, bias=False)
    
    # Add small regularization to ensure invertibility
    P_oo += 1e-6 * np.eye(n_obs)
    
    # Observation noise covariance (diagonal)
    R = observation_variance * np.eye(n_obs)
    
    # Kalman gain
    try:
        S = P_oo + R
        K = np.linalg.solve(S, P_oo.T)  # K is n_obs x n_obs
    except np.linalg.LinAlgError:
        # Fallback: simple nudging
        result[:, observed_indices] += 0.5 * obs_innov
        return _finite_clip(result)
    
    # Apply localization: Gaspari-Cohn taper on cross-covariance
    # Build localization matrix for all pairs of variables
    # We need to localize the cross-covariance P_xo (n_vars x n_obs)
    # For each variable i and observed variable j, apply GC based on distance
    
    # Compute distances in the Lorenz-96 ring (periodic)
    all_indices = np.arange(n_vars)
    obs_idx = np.array(observed_indices)
    
    # Distance matrix: n_vars x n_obs
    dist = np.abs(all_indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)  # Periodic distance
    
    # Localization scale (in indices)
    loc_scale = 5.0  # Typical for L96
    
    # GC weights
    gc_weights = _gaspari_cohn(dist / loc_scale, b=1.0, c=2.0)
    
    # Apply localization to the cross-covariance
    # P_xo is n_vars x n_obs
    P_xo = anomalies @ anomalies[:, observed_indices].T / (n_members - 1)
    
    # Localize: multiply each column j of P_xo by gc_weights[:, j]
    P_xo_loc = P_xo * gc_weights
    
    # Update all ensemble members
    # delta_x_i = P_xo_loc @ K @ (obs - x_i[obs])
    obs_diff = observations[None, :] - ensemble[:, observed_indices]  # n_members x n_obs
    updates = obs_diff @ K.T  # n_members x n_obs
    delta_x = updates @ P_xo_loc.T  # n_members x n_vars
    
    # Apply update
    result = ensemble + delta_x
    
    # Adaptive inflation: check if ensemble spread is too small
    # Compute spread at observed locations
    ens_spread_obs = np.std(ensemble[:, observed_indices], axis=0)
    obs_std = np.sqrt(observation_variance)
    
    # If spread is too small relative to observation error, inflate
    inflation_factor = 1.0
    for k in range(n_obs):
        if ens_spread_obs[k] < 0.5 * obs_std:
            inflation_factor = max(inflation_factor, 1.02)
    
    if inflation_factor > 1.0:
        # Inflate around the mean
        result = ens_mean + inflation_factor * (result - ens_mean)
    
    # Clip to valid range
    result = _finite_clip(result)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
