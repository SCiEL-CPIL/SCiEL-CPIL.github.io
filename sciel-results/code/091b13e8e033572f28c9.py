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


def _gaspari_cohn_weights(dist, L):
    """Compute Gaspari-Cohn localization weights."""
    # dist: array of distances
    # L: correlation length
    # Returns weights in [0, 1]
    x = dist / L
    w = np.zeros_like(x)
    mask1 = x <= 1.0
    mask2 = (x > 1.0) & (x <= 2.0)
    
    # Polynomial 1 for x in [0, 1]
    w[mask1] = (-0.25 * x[mask1]**5 + 0.5 * x[mask1]**4 + 0.625 * x[mask1]**3 - 0.5 * x[mask1]**2)
    
    # Polynomial 2 for x in (1, 2]
    w[mask2] = (0.25 * x[mask2]**5 - 2.5 * x[mask2]**4 + 10.0 * x[mask2]**3 - 15.0 * x[mask2]**2 + 10.0 * x[mask2] - 2.5)
    
    # Ensure non-negative
    w = np.maximum(w, 0.0)
    return w


def _compute_localization_matrix(D, observed_indices, L):
    """
    Compute localization weights matrix of shape (D, M).
    D = state dimension, M = number of observations.
    Uses minimum-image distance on the ring.
    """
    M = len(observed_indices)
    weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = _gaspari_cohn_weights(dist, L)
    return weights


def _enkf_update(ensemble, obs_vec, obs_idx, obs_var, L_loc, inflation_factor=1.0):
    """
    Standard EnKF update with Gaspari-Cohn localization.
    """
    N, D = ensemble.shape
    M = len(obs_idx)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Apply inflation to anomalies
    if abs(inflation_factor - 1.0) > 1e-6:
        anomalies = inflation_factor * anomalies
    
    obs_anomalies = anomalies[:, obs_idx]
    
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    # Regularize S for numerical stability
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    loc_mat = _compute_localization_matrix(D, obs_idx, L_loc)
    
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K = PHt_loc @ S_inv
    
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    deltas = innovations @ K.T
    
    updated_ensemble = ensemble + deltas
    
    return updated_ensemble


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_idx = np.asarray(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Adaptive Inflation
    # Calculate spread at observed indices
    obs_spread_pre = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_ens_spread = np.mean(obs_spread_pre)
    
    spread_ratio = mean_ens_spread / obs_noise_std if obs_noise_std > 0 else 1.0
    
    # Target spread ratio ~ 1.0
    # If spread is low, inflate. If high, deflate slightly.
    if spread_ratio < 1.0:
        inflation = 1.0 + 0.3 * (1.0 - spread_ratio)
    elif spread_ratio > 1.5:
        inflation = 1.0 - 0.1 * (spread_ratio - 1.5)
    else:
        inflation = 1.0
        
    inflation = max(0.95, min(1.3, inflation))
    
    # Pass 1: Narrow localization
    L_narrow = 2.0
    ens_pass1 = _enkf_update(ensemble, obs_vec, obs_idx, observation_variance, L_narrow, inflation)
    
    # Check time
    if time.time() > deadline:
        return np.clip(np.nan_to_num(ens_pass1, nan=0.0, posinf=100.0, neginf=-100.0), -100.0, 100.0)
        
    # Pass 2: Wide localization
    L_wide = 4.0
    # No additional inflation in pass 2, or very small
    updated_ensemble = _enkf_update(ens_pass1, obs_vec, obs_idx, observation_variance, L_wide, 1.0)
    
    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
