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


def _localize_covariance(P, D, observed_indices, loc_length):
    """
    Apply minimum-image distance localization to a (D, M) cross-covariance
    or (D, D) covariance matrix using ring geometry.
    """
    M = P.shape[1]
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L = loc_length
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return P * loc_weights


def _enkf_update(ensemble, obs_vec, observed_indices, observation_variance,
                 loc_length=4.0, inflation=1.0, blend_weight=1.0):
    """
    Single-pass Ensemble Kalman Filter with:
    - Regression-based (EnSRF) gain
    - Minimum-image distance localization on the ring
    - Optional multiplicative inflation
    - Optional blending with prior (blend_weight < 1.0 means partial update)
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # P_obs = cov of observed anomalies
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    # Cross-covariance P_xH^T: (D, M)
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Localize cross-covariance
    PHt_loc = _localize_covariance(PHt, D, observed_indices, loc_length)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Per-member innovation and update
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # delta_x_i = PHt_loc @ S_inv @ innovation_i
    Si_innov = innovations @ S_inv.T  # (N, M)
    delta_x = Si_innov @ PHt_loc.T  # (N, D)
    
    # Partial update via blend_weight
    updated_ensemble = ensemble + blend_weight * delta_x
    
    # Inflation
    if abs(inflation - 1.0) > 1e-6:
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    
    # Safety
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def _adaptive_inflation_factor(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Compute adaptive inflation factor based on ensemble spread vs observation noise.
    Uses a smooth, bounded approach.
    """
    N, D = ensemble.shape
    ens_obs = ensemble[:, observed_indices]
    ens_spread = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread)
    
    ratio = mean_spread / obs_noise_std if obs_noise_std > 1e-12 else 1.0
    
    # Smooth inflation: if spread is too low, inflate; if too high, deflate slightly
    if ratio < 0.5:
        return 1.15
    elif ratio < 0.8:
        return 1.05
    elif ratio > 3.0:
        return 0.95
    elif ratio > 2.0:
        return 0.98
    return 1.0


def _dynamics_blending(ensemble, F, dt, n_steps, blend_alpha):
    """
    Propagate each ensemble member forward n_steps with RK4,
    then blend: result = (1-alpha)*current + alpha*propagated.
    This maintains dynamical consistency.
    """
    N, D = ensemble.shape
    result = ensemble.copy()
    
    for i in range(N):
        x = result[i].copy()
        for _ in range(n_steps):
            x = _rk4_step(x, F, dt)
        result[i] = (1.0 - blend_alpha) * result[i] + blend_alpha * x
    
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    return result


def _multi_pass_enkf(ensemble, obs_vec, observed_indices, observation_variance,
                     F, loc_schedule, inflation_schedule, blend_weights,
                     dt=0.01, n_dyn_steps=2, dyn_blend_alpha=0.2):
    """
    Multi-pass EKF with:
    - Narrow-to-wide localization schedule
    - Adaptive inflation between passes
    - Optional dynamics blending between passes
    """
    N, D = ensemble.shape
    result = ensemble.copy()
    
    for pass_idx in range(len(loc_schedule)):
        loc = loc_schedule[pass_idx]
        infl = inflation_schedule[pass_idx]
        bw = blend_weights[pass_idx]
        
        # EKF update
        result = _enkf_update(result, obs_vec, observed_indices, observation_variance,
                              loc_length=loc, inflation=infl, blend_weight=bw)
        
        # Dynamics blending between passes (not after last)
        if pass_idx < len(loc_schedule) - 1 and dyn_blend_alpha > 0:
            result = _dynamics_blending(result, F, dt, n_dyn_steps, dyn_blend_alpha)
            
            # Re-apply adaptive inflation after dynamics
            if pass_idx + 1 < len(inflation_schedule):
                pass  # inflation handled in next EKF call
    
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    return result


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
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0].copy()
    else:
        obs_vec = obs_arr.flatten().copy()
    
    # Infer F from the ensemble statistics (Lorenz-96 equilibrium ~ F for mean)
    # Use a robust estimate: mean of the ensemble
    ens_mean = np.mean(ensemble, axis=0)
    F_est = np.clip(np.mean(ens_mean), 4.0, 12.0)
    
    # Strategy: 3-pass EKF with narrowing-then-widening localization
    # Pass 1: Narrow localization (L=3), full update, strong inflation if needed
    # Pass 2: Medium localization (L=5), full update, adaptive inflation
    # Pass 3: Wide localization (L=7), partial update (blend=0.5), light inflation
    
    # Compute adaptive inflation for pass 1
    infl1 = _adaptive_inflation_factor(ensemble, obs_vec, observed_indices, observation_variance)
    
    # Pass 1: Narrow localization
    result = _enkf_update(ensemble, obs_vec, observed_indices, observation_variance,
                          loc_length=3.0, inflation=infl1, blend_weight=1.0)
    
    if time.time() > deadline - 10.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Dynamics blending after pass 1
    result = _dynamics_blending(result, F_est, 0.01, 2, 0.25)
    
    if time.time() > deadline - 8.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Pass 2: Medium localization, adaptive inflation
    infl2 = _adaptive_inflation_factor(result, obs_vec, observed_indices, observation_variance)
    result = _enkf_update(result, obs_vec, observed_indices, observation_variance,
                          loc_length=5.0, inflation=infl2, blend_weight=1.0)
    
    if time.time() > deadline - 6.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Pass 3: Wide localization, partial update for smoothing
    infl3 = 1.0  # No additional inflation on final pass
    result = _enkf_update(result, obs_vec, observed_indices, observation_variance,
                          loc_length=7.0, inflation=infl3, blend_weight=0.5)
    
    if time.time() > deadline - 4.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Final dynamics consistency check: very light propagation blend
    result = _dynamics_blending(result, F_est, 0.01, 1, 0.1)
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
