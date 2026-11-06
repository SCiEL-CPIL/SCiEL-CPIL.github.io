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
            weights[d, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return weights


def _enkf_update_single(ensemble, obs_vec, obs_idx, obs_var, loc_width):
    """
    Perform a single EnKF update with given localization width.
    """
    N, D = ensemble.shape
    M = len(obs_idx)
    
    # Compute localization matrix
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Observation anomalies
    obs_anomalies = anomalies[:, obs_idx]
    
    # Observation covariance
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    # Regularize S to ensure invertibility
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
        
    # Cross-covariance between state and observations, localized
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # Gain matrix
    K = PHt_loc @ S_inv
    
    # Innovations: obs - ensemble_obs
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    
    # Update ensemble
    deltas = innovations @ K.T
    updated = ensemble + deltas
    
    return updated


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
    deadline = start_time + 115.0  # Safety margin
    
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
    
    if time.time() > deadline - 2.0:
        return ensemble.copy()
        
    # Two-pass EnKF strategy
    # Pass 1: Narrow localization to handle local dynamics and break spurious correlations
    L_narrow = 2.5
    updated_pass1 = _enkf_update_single(ensemble, obs_vec, obs_idx, observation_variance, L_narrow)
    
    if time.time() > deadline - 1.0:
        return np.clip(updated_pass1, -100.0, 100.0)
        
    # Pass 2: Wider localization to propagate information to unobserved neighbors
    L_wide = 4.0
    updated_pass2 = _enkf_update_single(updated_pass1, obs_vec, obs_idx, observation_variance, L_wide)
    
    # Adaptive Inflation
    # Compute ensemble spread in observation space before and after update
    # We want to maintain a certain ratio of spread to noise.
    # If spread is too low, inflate.
    
    ens_mean_orig = np.mean(ensemble, axis=0)
    obs_spread_orig = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    mean_spread_orig = np.mean(obs_spread_orig)
    
    ens_mean_new = np.mean(updated_pass2, axis=0)
    obs_spread_new = np.std(updated_pass2[:, obs_idx], axis=0, ddof=1)
    mean_spread_new = np.mean(obs_spread_new)
    
    noise_std = np.sqrt(observation_variance)
    
    # Target spread should be somewhat larger than noise std
    # If current spread is much smaller than noise, we likely collapsed.
    # Use a smooth inflation factor.
    # ratio = spread / noise_std
    # If ratio < 1, inflate.
    
    if mean_spread_new > 1e-8:
        ratio = mean_spread_new / noise_std
        # Desired ratio is around 1.5 or so. 
        # Inflation factor = desired / current
        # But we don't want to over-inflate if it's already good.
        # Smooth function: if ratio < 1, inflate by (1/ratio)^alpha
        # Let's use a simple heuristic:
        # If spread is less than noise_std, inflate by a factor that brings it closer.
        
        # A common approach:
        # inflation = 1.0 + alpha * max(0, 1.0 - ratio)
        # Or multiplicative:
        # inflation = 1.0 if ratio >= 1.0 else (1.0 / ratio)**0.5
        
        # Let's use a smooth multiplicative inflation based on the ratio.
        # We want to avoid sudden jumps.
        # If ratio is 0.5, inflate by 1.4 (approx 1/sqrt(0.5))
        # If ratio is 1.0, inflate by 1.0
        # If ratio is 2.0, inflate by 1.0 (or slightly less to prevent growth?)
        
        # Let's stick to a conservative inflation that only acts when spread is low.
        # inflation = 1.0 + 0.2 * np.exp(-ratio)
        # This adds a small amount of inflation that decays as ratio increases.
        
        inflation = 1.0 + 0.1 * np.exp(-1.5 * ratio)
        
        # Apply inflation around the new mean
        updated_final = ens_mean_new + inflation * (updated_pass2 - ens_mean_new)
    else:
        # If spread collapsed to zero, re-inflate significantly
        updated_final = ens_mean_new + 1.5 * noise_std * np.random.randn(N, D) # Fallback, but shouldn't happen often
        
    # Final safety checks
    updated_final = np.nan_to_num(updated_final, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_final = np.clip(updated_final, -100.0, 100.0)
    
    return updated_final

# EVOLVE-BLOCK-END
