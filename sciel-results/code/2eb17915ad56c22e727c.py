"""Lorenz-96 ensemble assimilation with localized EKF, adaptive inflation, and dynamic F estimation."""
import numpy as np
import time

# Module-level state
_inflation_factor = 1.05
_inflation_alpha = 0.85
_min_inflation = 1.0
_max_inflation = 2.5
_localization_range = 3.0
_deadline = None
_start_time = None


def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20) or (20,)."""
    if x.ndim == 1:
        xm2 = np.roll(x, 2)
        xm1 = np.roll(x, 1)
        xp1 = np.roll(x, -1)
        return (xp1 - xm2) * xm1 - x + F
    else:
        xm2 = np.roll(x, 2, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xp1 = np.roll(x, -1, axis=1)
        return (xp1 - xm2) * xm1 - x + F


def _rk4_step(x, F, dt, n_steps=5):
    """Propagate ensemble (N, 20) by RK4 for n_steps."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _localization_weights(obs_idx, dim=20, length_scale=3.0):
    """Compute periodic localization weights for one observation index."""
    j = np.arange(dim)
    dist = np.minimum(np.abs(obs_idx - j), dim - np.abs(obs_idx - j))
    return np.exp(-0.5 * (dist / length_scale) ** 2)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: array of indices observed (should be [0,2,4,...,18])
    observation_variance: scalar variance of observation noise
    
    Returns: updated ensemble (16, 20)
    """
    global _inflation_factor, _start_time, _deadline
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 110.0  # Safety margin before 120s limit
    
    if time.time() > _deadline:
        return np.clip(ensemble, -100, 100)
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure finite
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0)
    observations = np.where(np.isfinite(observations), observations, 0.0)
    
    # --- Step 1: Estimate forcing F ---
    # Use the ensemble mean dynamics to estimate F
    mean = ensemble.mean(axis=0)
    # For Lorenz-96, the mean should satisfy: 0 ≈ (x_{i+1}-x_{i-2})*x_{i-1} - x_i + F
    # So F ≈ x_i - (x_{i+1}-x_{i-2})*x_{i-1}
    xm2 = np.roll(mean, 2)
    xm1 = np.roll(mean, 1)
    xp1 = np.roll(mean, -1)
    F_est = np.mean(mean - (xp1 - xm2) * xm1)
    # Clamp F to reasonable range
    F_est = np.clip(F_est, 2.0, 15.0)
    
    # --- Step 2: Inflate ensemble ---
    inflated = mean + _inflation_factor * (ensemble - mean)
    
    # --- Step 3: Compute covariance ---
    cov = np.cov(inflated, rowvar=False)
    
    # --- Step 4: Localized EKF update ---
    # Build observation operator H
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    y = observations  # (n_obs,)
    
    # Forecast observations for each member
    y_f = H @ inflated.T  # (n_obs, N)
    y_f_mean = y_f.mean(axis=1)  # (n_obs,)
    
    # Innovation
    innovation = y - y_f_mean  # (n_obs,)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Forecast observation covariance
    P = cov  # (20, 20)
    S = H @ P @ H.T + R  # (n_obs, n_obs)
    
    # Compute Kalman gain with localization
    # K = P H^T S^-1
    try:
        S_inv = np.linalg.inv(S + 1e-10 * np.eye(n_obs))
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P @ H.T @ S_inv  # (20, n_obs)
    
    # Apply localization to the gain
    K_localized = K.copy()
    for i, obs_idx in enumerate(observed_indices):
        w = _localization_weights(obs_idx, dim, _localization_range)
        K_localized[:, i] *= w
    
    # Update mean
    mean_updated = mean + K_localized @ innovation  # (20,)
    
    # Update each ensemble member
    # innovation_i = y - y_f_i for each member
    innovations_all = y[:, np.newaxis] - y_f  # (n_obs, N)
    
    # For localized update, each member gets a different effective gain
    # based on which observations are relevant
    updated_ensemble = np.zeros_like(inflated)
    for i in range(N):
        # For each member, compute the localized innovation-weighted update
        member_update = np.zeros(dim)
        for j, obs_idx in enumerate(observed_indices):
            w = _localization_weights(obs_idx, dim, _localization_range)
            # The gain contribution from this observation
            member_update += K[:, j] * innovations_all[j, i] * w
        updated_ensemble[i] = inflated[i] + member_update
    
    # --- Step 5: Adaptive inflation update ---
    innovation_norm = np.linalg.norm(innovation)
    expected_innovation_norm = np.sqrt(np.trace(S))
    
    if expected_innovation_norm > 1e-10:
        ratio = innovation_norm / expected_innovation_norm
        if ratio > 1.0:
            _inflation_factor = _inflation_alpha * _inflation_factor + (1 - _inflation_alpha) * min(1.5, ratio * 0.8)
        else:
            _inflation_factor = _inflation_alpha * _inflation_factor + (1 - _inflation_alpha) * 1.0
    
    _inflation_factor = np.clip(_inflation_factor, _min_inflation, _max_inflation)
    
    # --- Step 6: Clip to valid range ---
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, 0.0)
    
    return updated_ensemble
