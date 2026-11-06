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

def _gaspari_cohn(d, L):
    """
    Gaspari-Cohn 5th order localization function.
    d: distance
    L: correlation length scale
    """
    x = d / L
    # np.where is vectorized
    # 0 <= x <= 1
    q1 = (2/3)*x**5 - (5/2)*x**4 + (7/2)*x**3 - (2/3)*x**2 - 1
    # 1 < x <= 2
    q2 = (2/3)*x**5 - 5*x**4 + (5/2)*x**3 + 4
    # x > 2 is 0
    
    # Handle boundaries carefully
    # x is an array
    mask1 = (x >= 0) & (x <= 1)
    mask2 = (x > 1) & (x <= 2)
    
    res = np.zeros_like(x)
    res[mask1] = q1[mask1]
    res[mask2] = q2[mask2]
    
    # Ensure non-negative (numerical errors might make it slightly negative near 2)
    res = np.maximum(res, 0.0)
    return res

def _compute_localization_matrix_gc(D, observed_indices, L):
    """
    Compute localization weights matrix of shape (D, M) using Gaspari-Cohn.
    D = state dimension, M = number of observations.
    """
    M = len(observed_indices)
    weights = np.zeros((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = _gaspari_cohn(np.array([dist]), L)[0]
    return weights

def _enkf_update_localized(ensemble, obs_vec, obs_idx, obs_var, L_loc):
    """
    Perform a Localized EnKF update.
    """
    N, D = ensemble.shape
    M = len(obs_idx)
    
    # Compute localization matrix
    loc_mat = _compute_localization_matrix_gc(D, obs_idx, L_loc)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Observation anomalies (N, M)
    obs_anomalies = anomalies[:, obs_idx]
    
    # Observation covariance P_obs (M, M)
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    
    # Innovation covariance S (M, M)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    # Regularize S
    # Add small jitter for stability
    S += 1e-6 * np.eye(M)
    
    # Cross-covariance P_xy (D, M)
    P_xy = (anomalies.T @ obs_anomalies) / (N - 1)
    
    # Localize the cross-covariance
    # The localization is applied to the background error covariance P_xx.
    # In the gain calculation K = P_xy S^-1, the localization effectively scales P_xy.
    # Standard practice: K_loc = (P_xy * Loc) S^-1
    P_xy_loc = P_xy * loc_mat
    
    # Gain K (D, M)
    # Solve K S = P_xy_loc  => K = P_xy_loc S^-1
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K = P_xy_loc @ S_inv
    
    # Innovations (N, M)
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    
    # Update ensemble
    deltas = innovations @ K.T
    updated = ensemble + deltas
    
    return updated

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
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
        
    # Strategy: Localized EnKF with Gaspari-Cohn localization
    # Single pass with tuned width
    # For D=20, L=3.0 (range 6) covers most of the ring effectively with GC
    # Let's try L=2.5 for GC, which has effective range 5.
    
    L_gc = 2.5
    
    updated_ensemble = _enkf_update_localized(ensemble, obs_vec, obs_idx, observation_variance, L_gc)
    
    if time.time() > deadline - 1.0:
        return np.clip(updated_ensemble, -100.0, 100.0)
        
    # Adaptive Inflation
    # Calculate spread in observation space
    ens_mean_new = np.mean(updated_ensemble, axis=0)
    obs_spread_new = np.std(updated_ensemble[:, obs_idx], axis=0, ddof=1)
    mean_spread_new = np.mean(obs_spread_new)
    
    noise_std = np.sqrt(observation_variance)
    
    # Target ratio: We want the ensemble spread to be slightly larger than the noise
    # to account for model error and to prevent collapse.
    # A ratio of 1.2 to 1.5 is typical.
    target_ratio = 1.3
    
    if mean_spread_new > 1e-8:
        current_ratio = mean_spread_new / noise_std
        # If current ratio is less than target, inflate
        if current_ratio < target_ratio:
            # Smooth inflation factor
            # factor = target_ratio / current_ratio
            # To avoid huge jumps, we can dampen it
            inflation = 1.0 + 0.5 * (target_ratio - current_ratio) / target_ratio
            # Or multiplicative:
            # inflation = (target_ratio / current_ratio) ** 0.5
            # Let's use a gentle multiplicative approach
            inflation = np.sqrt(target_ratio / current_ratio)
            
            # Cap inflation to prevent divergence
            inflation = min(inflation, 1.5)
            
            updated_ensemble = ens_mean_new + inflation * (updated_ensemble - ens_mean_new)
        else:
            # If spread is too high, we might deflate slightly?
            # Usually we don't deflate in EnKF, we let the update handle it.
            # But if it's very high, it might indicate a bad localization.
            # Let's leave it alone to be safe.
            pass
    else:
        # Fallback if collapsed
        updated_ensemble = ens_mean_new + 1.5 * noise_std * np.random.randn(N, D)
        
    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
