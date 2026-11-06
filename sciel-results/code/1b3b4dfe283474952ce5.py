"""Stochastic Ensemble Kalman Filter with localization and inflation for Lorenz-96 assimilation."""
import numpy as np
import time

def _lorenz96_step(x, F, dt=0.01, n=20):
    """Single RK4 step for Lorenz-96."""
    def f(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _localize_covariance(ensemble, observed_indices, n_vars=20, width=5.0):
    """
    Apply spatial localization to the ensemble covariance.
    Uses a Gaussian decay kernel based on spatial distance with periodic boundary.
    """
    n_ens = ensemble.shape[0]
    n_obs = len(observed_indices)
    
    # Compute background ensemble mean and anomalies
    bg_mean = ensemble.mean(axis=0)
    anomalies = ensemble - bg_mean  # (n_ens, n_vars)
    
    # Compute sample covariance in observation space
    # C_oo = (1/(n_ens-1)) * A_o @ A_o.T
    A_obs = anomalies[:, observed_indices]  # (n_ens, n_obs)
    C_oo = (A_obs @ A_obs.T) / (n_ens - 1)  # (n_obs, n_obs)
    
    # Apply localization to C_oo
    obs_idx = np.array(observed_indices)
    for i in range(n_obs):
        for j in range(n_obs):
            dist = abs(obs_idx[i] - obs_idx[j])
            dist = min(dist, n_vars - dist)  # periodic
            C_oo[i, j] *= np.exp(-0.5 * (dist / width) ** 2)
    
    # Ensure symmetry
    C_oo = 0.5 * (C_oo + C_oo.T)
    
    return C_oo, A_obs, bg_mean, anomalies

def _compute_gain(C_oo, obs_var, n_ens, n_obs):
    """
    Compute the Kalman gain using the ensemble-based formulation.
    Uses the perturbed observation operator approach.
    """
    # Add observation noise to C_oo
    R = obs_var * np.eye(n_obs)
    S = C_oo + R  # Innovation covariance
    
    # Invert S with regularization
    # Use SVD for numerical stability
    try:
        U, s, Vt = np.linalg.svd(S)
        # Regularize small singular values
        s_reg = np.maximum(s, 1e-10)
        S_inv = Vt.T @ np.diag(1.0 / s_reg) @ U.T
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Gain: K = C_xo @ S^{-1}
    # We'll compute this in the analyze function where we have C_xo
    return S_inv

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
    observations : ndarray, shape (10,)
    observed_indices : array_like, shape (10,)
    observation_variance : float
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    t0 = time.time()
    deadline = t0 + 100.0  # Safety margin before 120s limit
    
    n_ens, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # --- Step 1: Compute background statistics ---
    bg_mean = ensemble.mean(axis=0)
    anomalies = ensemble - bg_mean  # (n_ens, n_vars)
    
    # --- Step 2: Localized observation-space covariance ---
    A_obs = anomalies[:, obs_idx]  # (n_ens, n_obs)
    C_oo = (A_obs @ A_obs.T) / (n_ens - 1)
    
    # Apply Gaussian localization
    loc_width = 5.0
    for i in range(n_obs):
        for j in range(n_obs):
            dist = abs(obs_idx[i] - obs_idx[j])
            dist = min(dist, n_vars - dist)
            C_oo[i, j] *= np.exp(-0.5 * (dist / loc_width) ** 2)
    C_oo = 0.5 * (C_oo + C_oo.T)
    
    # --- Step 3: Innovation covariance and its inverse ---
    R = observation_variance * np.eye(n_obs)
    S = C_oo + R
    
    # Regularized inverse via SVD
    try:
        U, s, Vt = np.linalg.svd(S)
        s_reg = np.maximum(s, 1e-8)
        S_inv = Vt.T @ np.diag(1.0 / s_reg) @ U.T
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # --- Step 4: Cross-covariance C_xo ---
    C_xo = (anomalies @ A_obs.T) / (n_ens - 1)  # (n_vars, n_obs)
    
    # --- Step 5: Kalman gain ---
    K = C_xo @ S_inv  # (n_vars, n_obs)
    
    # --- Step 6: Stochastic EnKF update ---
    # Inject observation noise into the mean update (StEnKF)
    # The stochastic part: add sqrt(obs_var) * N(0, I) to observations
    # But we use the standard StEnKF formulation:
    # x_a = x_b + K (y - H x_b) + K * sqrt(R) * eps
    # where eps ~ N(0, I)
    
    innovations = observations - A_obs.mean(axis=0)  # (n_obs,)
    
    # Stochastic perturbation
    eps = np.random.randn(n_obs)
    stoch_term = K @ (np.sqrt(observation_variance) * eps)
    
    # Update each ensemble member
    result = np.empty_like(ensemble)
    for m in range(n_ens):
        # Deterministic part
        innov_m = observations - ensemble[m, obs_idx]
        result[m] = ensemble[m] + K @ innov_m + stoch_term
    
    # --- Step 7: Adaptive multiplicative inflation ---
    # Check if ensemble spread is adequate
    ens_mean = result.mean(axis=0)
    ens_anom = result - ens_mean
    spread_obs = np.sqrt(np.mean(ens_anom[:, obs_idx] ** 2, axis=0))
    obs_std = np.sqrt(observation_variance)
    
    # If spread is too low, inflate
    ratio = spread_obs / obs_std
    if np.any(ratio < 0.8):
        # Inflate anomalies multiplicatively
        inflation_factor = 1.0 + 0.05 * np.maximum(0, 0.8 - ratio).max()
        result = ens_mean + inflation_factor * ens_anom
    
    # --- Step 8: Clamp to valid range ---
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() > deadline:
        return result
    
    return result
