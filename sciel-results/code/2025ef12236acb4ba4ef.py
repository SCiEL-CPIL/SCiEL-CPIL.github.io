"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

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


def _gaussian_localization(D, observed_indices, length_scale):
    """
    Compute Gaussian localization weights for cross-covariance.
    Returns (D, M) array where M = len(observed_indices).
    """
    M = len(observed_indices)
    loc = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc[d, m_idx] = np.exp(-0.5 * (dist / length_scale) ** 2)
    return loc


def _esrf_update(ensemble, obs_vec, observed_indices, observation_variance,
                 localization_length, inflation_factor):
    """
    Single-pass Ensemble Square Root Filter with localization.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    obs_vec = np.asarray(obs_vec).flatten()
    if len(obs_vec) != M:
        if len(obs_vec) > M:
            obs_vec = obs_vec[:M]
        else:
            tmp = np.zeros(M)
            tmp[:len(obs_vec)] = obs_vec
            obs_vec = tmp
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    obs_anoms = anomalies[:, observed_indices]  # (N, M)
    
    # Innovation covariance in observation space
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    # Cross-covariance
    PHt = (anomalies.T @ obs_anoms) / (N - 1)  # (D, M)
    
    # Localization
    loc = _gaussian_localization(D, observed_indices, localization_length)
    PHt_loc = PHt * loc
    
    # Solve for gain
    try:
        L = np.linalg.cholesky(S)
        # K = PHt_loc @ S^{-1}
        # S^{-1} = L^{-T} @ L^{-1}
        tmp = np.linalg.solve(L.T, PHt_loc.T)  # (M, D)
        K = np.linalg.solve(L, tmp).T  # (D, M)
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
            K = PHt_loc @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = PHt_loc @ S_inv
    
    # Apply update
    obs_ens = ensemble[:, observed_indices]
    innovations = obs_vec[None, :] - obs_ens  # (N, M)
    delta_x = innovations @ K.T  # (N, D)
    
    updated = ensemble + delta_x
    
    # Apply multiplicative inflation about the mean
    if inflation_factor != 1.0:
        updated = ens_mean + inflation_factor * (updated - ens_mean)
    
    return updated


def _compute_adaptive_inflation(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Compute adaptive multiplicative inflation factor based on
    ensemble spread vs observation innovation.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    ens_mean = np.mean(ensemble, axis=0)
    obs_anoms = (ensemble - ens_mean)[:, observed_indices]
    
    # Ensemble variance in observation space
    ens_var_obs = np.mean(np.var(obs_anoms, axis=0, ddof=1))
    
    # Observation variance
    obs_var = observation_variance
    
    # Innovation variance (should be ~ ens_var + obs_var if well-calibrated)
    innovations = obs_vec - (ensemble[:, observed_indices])
    innov_var = np.mean(np.var(innovations, axis=0, ddof=1))
    
    # Expected innovation variance
    expected_innov_var = ens_var_obs + obs_var
    
    # Inflation factor: if spread is too low relative to innovations, inflate
    if expected_innov_var < 1e-10:
        return 1.0
    
    ratio = innov_var / expected_innov_var
    
    # Target ratio ~ 1.0. If ratio > 1, ensemble is under-dispersed.
    # Use a smooth adjustment
    if ratio > 1.05:
        # Under-dispersed: inflate
        # inflation = sqrt(ratio) would be too aggressive; use a damped version
        inflation = 1.0 + 0.5 * (ratio - 1.0)
        inflation = min(inflation, 1.3)
    elif ratio < 0.95:
        # Over-dispersed: slight deflation
        inflation = 1.0 - 0.3 * (1.0 - ratio)
        inflation = max(inflation, 0.95)
    else:
        inflation = 1.0
    
    return inflation


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
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    if len(obs_vec) != M:
        if len(obs_vec) > M:
            obs_vec = obs_vec[:M]
        else:
            tmp = np.zeros(M)
            tmp[:len(obs_vec)] = obs_vec
            obs_vec = tmp
    
    # Two-pass localized EnKF: narrow then wide
    # Pass 1: narrow localization to correct local errors
    result1 = _esrf_update(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=2.0, inflation_factor=1.0
    )
    
    # Compute adaptive inflation from pass 1 result
    inflation = _compute_adaptive_inflation(result1, obs_vec, observed_indices, observation_variance)
    
    # Pass 2: wider localization with inflation to correct global structure
    result2 = _esrf_update(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=4.0, inflation_factor=inflation
    )
    
    # Blend the two passes: weight toward the wider pass but keep narrow corrections
    # Use 0.5/0.5 blend for stability
    result = 0.5 * result1 + 0.5 * result2
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 145.0:
        return result
    
    return result

# EVOLVE-BLOCK-END
