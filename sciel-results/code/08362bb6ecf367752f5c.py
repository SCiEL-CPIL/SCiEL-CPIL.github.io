"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 115.0  # Safety margin before 120s evaluator limit
_START_TIME = [time.time()]  # Mutable container to track start time across calls


def _elapsed():
    """Return elapsed time since first call."""
    return time.time() - _START_TIME[0]


def _remaining():
    """Return remaining time budget."""
    return _DEADLINE_SECONDS - _elapsed()


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    n = len(x)
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


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _compute_localization_matrix(D, observed_indices, localization_length):
    """Compute localization weight matrix (D, M) for ring geometry."""
    M = len(observed_indices)
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    return loc_weights


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance,
                            localization_length=3.0, inflation_base=1.02,
                            max_inflation=1.3):
    """
    Ensemble Kalman Filter update with Gaussian localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    inflation_base : float - base multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute observation-space covariance P_obs
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute cross-covariance PHt
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Apply localization to the cross-covariance
    loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Solve for S_inv using Cholesky for stability
    try:
        L_S = np.linalg.cholesky(S)
        # S_inv = inv(L) @ inv(L.T)
        # Instead, solve L @ Y = I, then S_inv = Y @ Y.T
        Y = np.linalg.solve(L_S, np.eye(M))
        S_inv = Y.T @ Y
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    # Compute updated ensemble using square-root filter approach
    # For each member: delta_x_m = PHt_loc @ S_inv @ (obs_vec - x_m[obs])
    # Vectorized: innovations = obs_vec[None, :] - ensemble[:, observed_indices]  (N, M)
    # Si_innov = innovations @ S_inv.T  (N, M)
    # delta = Si_innov @ PHt_loc.T  (N, D)
    
    innovations = obs_vec[None, :] - ensemble[:, observed_indices]  # (N, M)
    Si_innov = innovations @ S_inv.T  # (N, M)
    delta = Si_innov @ PHt_loc.T  # (N, D)
    
    updated_ensemble = ensemble + delta
    
    # Adaptive inflation based on innovation spread
    # Compute innovation for the ensemble mean
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    innov_mean = obs_vec - obs_mean_pre  # (M,)
    
    # Expected innovation variance from ensemble
    P_obs_diag = np.diag(P_obs)
    R_diag = np.full(M, observation_variance)
    expected_innov_var = P_obs_diag + R_diag  # (M,)
    expected_innov_std = np.sqrt(np.maximum(expected_innov_var, 1e-10))
    
    # Actual innovation magnitude
    actual_innov_abs = np.abs(innov_mean)
    
    # If innovations are larger than expected, the ensemble is under-dispersed
    # Compute a ratio and use it to determine inflation
    # Use a smooth, bounded inflation factor
    innov_ratio = np.mean(actual_innov_abs / expected_innov_std)
    
    # Map innovation ratio to inflation factor
    # If ratio > 1, we need inflation
    # Use a smooth mapping: inflation = 1 + alpha * max(0, ratio - 1)
    alpha = 0.5
    inflation = 1.0 + alpha * max(0.0, innov_ratio - 1.0)
    inflation = min(inflation, max_inflation)
    
    # Also apply a small base inflation to counteract filter degeneracy
    inflation = max(inflation, inflation_base)
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
        updated_ensemble = ens_mean + inflation * (updated_ensemble - ens_mean)
    
    # Ensure finite values and clip to valid range
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
    # Deadline guard
    if _remaining() < 0.5:
        # Not enough time; return ensemble as-is (safe fallback)
        return np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        # If it's (N, M), assume all rows are the same or take the first one
        # Standard EnKF uses a single observation vector
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Check time again before the main computation
    if _remaining() < 1.0:
        return np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # EKF with localization and adaptive inflation
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.0,
        inflation_base=1.02,
        max_inflation=1.3
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
