"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with optimal gain (regression-based) and adaptive inflation.
    Uses minimum-image distance localization on the ring.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    # Ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered anomalies
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observed anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute P_obs = cov(observed) + R
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    # Cross-covariance P_xH^T
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Localization: minimum-image distance on ring
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L = 5.0  # localization length scale
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Compute S_inv
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Innovation for each member
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # Gain-like update: delta_x = PHt_loc @ S_inv @ innovation
    Si_innov = innovations @ S_inv.T  # (N, M)
    delta_x = Si_innov @ PHt_loc.T  # (N, D)
    
    updated_ensemble = ensemble + delta_x  # (N, D)
    
    # Adaptive inflation: compare ensemble spread to observation noise
    ens_spread_obs = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)
    
    if mean_spread < obs_noise_std * 0.7:
        inflation = 1.08
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    elif mean_spread > obs_noise_std * 2.0:
        inflation = 0.97
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    
    # Safety: clip and handle NaNs
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def _ensemble_smoother(ensemble, obs_vec, observed_indices, observation_variance, F, dt=0.01, n_steps=5):
    """
    Ensemble Kalman Smoother: propagate ensemble forward, then apply EKF update.
    This uses the fact that we can simulate the dynamics to get better state estimates.
    """
    N, D = ensemble.shape
    
    # Propagate each member forward n_steps with RK4
    propagated = np.empty_like(ensemble)
    for m in range(N):
        x = ensemble[m].copy()
        for _ in range(n_steps):
            x = _rk4_step(x, F, dt)
        propagated[m] = x
    
    # Apply EKF to the propagated ensemble
    return _ensemble_kalman_filter(propagated, obs_vec, observed_indices, observation_variance)


def _ensemble_kalman_smoother(ensemble, obs_vec, observed_indices, observation_variance, F, dt=0.01, n_steps=5):
    """
    Two-pass approach: first EKF on current ensemble, then propagate and do a second EKF.
    This smooths out inconsistencies from the first update.
    """
    N, D = ensemble.shape
    
    # First pass: EKF on current ensemble
    first_pass = _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance)
    
    # Propagate the updated ensemble forward
    propagated = np.empty_like(first_pass)
    for m in range(N):
        x = first_pass[m].copy()
        for _ in range(n_steps):
            x = _rk4_step(x, F, dt)
        propagated[m] = x
    
    # Second pass: EKF on propagated ensemble with wider localization
    # Use a modified EKF with wider localization for the second pass
    N, D = propagated.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    ens_mean = np.mean(propagated, axis=0)
    anomalies = propagated - ens_mean
    obs_anomalies = anomalies[:, observed_indices]
    
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = observation_variance * np.eye(M)
    S = P_obs + R
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    
    # Wider localization for second pass
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L2 = 7.0  # wider localization
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L2) ** 2)
    
    PHt_loc = PHt * loc_weights
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    ens_obs = propagated[:, observed_indices]
    innovations = obs_vec[None, :] - ens_obs
    Si_innov = innovations @ S_inv.T
    delta_x = Si_innov @ PHt_loc.T
    
    updated_ensemble = propagated + delta_x
    
    # Adaptive inflation
    ens_spread_obs = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)
    
    if mean_spread < obs_noise_std * 0.7:
        inflation = 1.08
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    elif mean_spread > obs_noise_std * 2.0:
        inflation = 0.97
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Basic EKF with localization and adaptive inflation
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance
    )
    
    # Deadline guard
    if time.time() > deadline:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
