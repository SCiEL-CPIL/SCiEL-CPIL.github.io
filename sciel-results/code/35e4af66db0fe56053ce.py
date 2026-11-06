"""Ensemble analysis for partial-observation Lorenz-96 assimilation.

Uses a Localized Ensemble Kalman Filter with:
  - Gaussian localization on cross-covariances
  - Adaptive multiplicative inflation (spread-based)
  - Ensemble adjustment transform (ET) to preserve covariance
  - Deadline guard for the 150 s wall-clock budget
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_DEADLINE_FRACTION = 0.88


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
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


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _periodic_distance(i, j, D):
    """Periodic distance on a ring of length D."""
    d = abs(i - j)
    return min(d, D - d)


def _compute_localization_matrix(D, observed_indices, L_loc):
    """Compute (D, M) localization weight matrix."""
    M = len(observed_indices)
    loc = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = _periodic_distance(d, obs_i, D)
            loc[d, m_idx] = _gaussian_localization(dist, L_loc)
    return loc


def _enkf_et_update(ensemble, obs_vec, observed_indices, R_diag, loc_weights):
    """
    Ensemble Kalman Filter update using the Ensemble Transform (ET) method.
    
    Parameters
    ----------
    ensemble : (N, D) array
    obs_vec : (M,) array of observations
    observed_indices : (M,) array of indices
    R_diag : (M,) array of observation variances
    loc_weights : (D, M) localization weights
    
    Returns
    -------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    Hx = ensemble[:, observed_indices]  # (N, M)
    Hx_mean = np.mean(Hx, axis=0)
    Hx_anom = Hx - Hx_mean  # (N, M)
    
    innovation = obs_vec - Hx_mean  # (M,)
    
    # Sample covariances
    P_oo = (Hx_anom.T @ Hx_anom) / (N - 1)  # (M, M)
    P_ox = (anomalies.T @ Hx_anom) / (N - 1)  # (D, M)
    
    # Localize cross-covariance
    P_ox_loc = P_ox * loc_weights  # (D, M)
    
    # Innovation covariance
    S = P_oo + np.diag(R_diag)  # (M, M)
    
    # Compute Kalman gain K = P_ox_loc S^{-1}  (D, M)
    try:
        L_chol = np.linalg.cholesky(S)
        tmp = np.linalg.solve(L_chol, P_ox_loc.T)
        K = np.linalg.solve(L_chol.T, tmp).T  # (D, M)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_ox_loc @ S_inv  # (D, M)
    
    # ET method: construct transform T such that
    #   x_a = T @ x  (row-wise), preserving ensemble mean
    # Standard ET:
    #   T = I - K H  (for the anomaly part)
    # But we need to be careful: the ET transform acts on the full state.
    # 
    # For the ET method (Hunt et al. 2007, Whitaker & Hamill 2012):
    #   Let A = anomalies (N, D)
    #   Let HA = Hx_anom (N, M)
    #   Innovation: y - Hx_mean (M,)
    #   K: (D, M)
    #   
    # The analysis anomaly for member i:
    #   A_a_i = A_i + K @ (y - Hx_mean - HA_i)
    #         = A_i + K @ innovation - K @ HA_i
    #
    # This is equivalent to:
    #   A_a = A + K @ (innovation[None,:] - HA)
    #       = A + K @ innovation[None,:] - A @ (HA.T @ K.T).T  -- wait
    #
    # Actually: K @ HA_i where HA_i is (M,) and K is (D, M)
    # K @ HA_i = (K @ HA_i)  which is (D,)
    # Vectorized: (N, M) @ (M, D).T -> no
    # K is (D, M), HA is (N, M)
    # K @ HA_i for each i: this is (D, M) @ (M,) = (D,)
    # For all i: (N, M) @ (M, D).T? No.
    # Actually: sum_m K[d,m] * HA[i,m] for each i,d
    # = (HA @ K.T)  where HA is (N,M), K.T is (M,D) -> (N, D). Yes.
    
    K_innov = K @ innovation  # (D,)
    HA_K = Hx_anom @ K.T  # (N, D)
    
    updated_ensemble = ensemble + K_innov[None, :] - HA_K
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using a Localized Ensemble Kalman Filter
    with adaptive multiplicative inflation.
    
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
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        return np.clip(ensemble, -100.0, 100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # --- Localized Ensemble Kalman Filter ---
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation operator
    Hx = ensemble[:, observed_indices]  # (N, M)
    Hx_mean = np.mean(Hx, axis=0)
    Hx_anom = Hx - Hx_mean  # (N, M)
    
    # Innovation
    innovation = obs_vec - Hx_mean  # (M,)
    
    # Covariances
    P_oo = (Hx_anom.T @ Hx_anom) / (N - 1)  # (M, M)
    P_ox = (anomalies.T @ Hx_anom) / (N - 1)  # (D, M)
    
    # Observation noise
    R_diag = observation_variance * np.ones(M)
    R = np.diag(R_diag)
    
    # Innovation covariance
    S = P_oo + R  # (M, M)
    
    # Localization
    L_loc = 2.5
    loc_weights = _compute_localization_matrix(D, observed_indices, L_loc)
    
    P_ox_loc = P_ox * loc_weights
    
    # Adaptive multiplicative inflation
    # Compare ensemble spread in observation space to observation noise
    # Expected innovation std from S diagonal
    expected_innov_std = np.sqrt(np.mean(np.diag(S)))
    innov_std = np.std(innovation)
    
    if expected_innov_std > 1e-10:
        ratio = innov_std / expected_innov_std
        # Damped inflation: gently adjust spread
        inflation_factor = np.clip(1.0 + 0.4 * (ratio - 1.0), 0.85, 1.6)
    else:
        inflation_factor = 1.0
    
    # Apply inflation to anomalies before computing gain
    inflated_anomalies = anomalies * inflation_factor
    P_ox_loc_inflated = (inflated_anomalies.T @ Hx_anom) / (N - 1) * loc_weights
    
    # Also inflate P_oo for the innovation covariance to account for spread
    # Actually, we should inflate the full covariance. Let's inflate P_oo too.
    inflated_Hx_anom = Hx_anom * inflation_factor
    P_oo_inflated = (inflated_Hx_anom.T @ inflated_Hx_anom) / (N - 1)
    S_inflated = P_oo_inflated + R
    
    # Kalman gain with inflated covariances
    try:
        L_chol = np.linalg.cholesky(S_inflated)
        tmp = np.linalg.solve(L_chol, P_ox_loc_inflated.T)
        K = np.linalg.solve(L_chol.T, tmp).T  # (D, M)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_inflated)
        K = P_ox_loc_inflated @ S_inv  # (D, M)
    
    # Update ensemble members
    K_innov = K @ innovation  # (D,)
    Hx_anom_K = Hx_anom @ K.T  # (N, D)
    
    updated_ensemble = ensemble + K_innov[None, :] - Hx_anom_K
    
    # Re-center to ensure mean consistency
    new_mean = np.mean(updated_ensemble, axis=0)
    expected_mean = ens_mean + K_innov
    updated_ensemble = updated_ensemble + (expected_mean - new_mean)
    
    # Safety
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
