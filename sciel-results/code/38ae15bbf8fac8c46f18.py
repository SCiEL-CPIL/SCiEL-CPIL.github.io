"""Lorenz-96 ensemble Kalman filter with localization and adaptive inflation."""
import numpy as np
import time

_DEADLINE = 115.0
_T0 = time.time()

def _lorenz96_deriv(state, F):
    """Lorenz-96 derivative."""
    n = len(state)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (state[(i+1) % n] - state[(i-2) % n]) * state[(i-1) % n] - state[i] + F
    return dx

def _rk4_step(state, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_deriv(state, F)
    k2 = _lorenz96_deriv(state + 0.5 * dt * k1, F)
    k3 = _lorenz96_deriv(state + 0.5 * dt * k2, F)
    k4 = _lorenz96_deriv(state + dt * k3, F)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _lorenz96_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    return _rk4_step(x, F, dt)

def _random_rotation_matrix(n, rng):
    """Generate a random rotation matrix using QR decomposition."""
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det=1)
    d = np.sign(np.diag(R))
    Q = Q * d
    return Q

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis step for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed variables
    - observed_indices: array of observed variable indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - Updated ensemble with same (16, 20) shape
    """
    N = ensemble.shape[0]  # number of ensemble members (16)
    M = ensemble.shape[1]  # number of variables (20)
    K = len(observed_indices)  # number of observed variables (10)
    
    result = ensemble.copy()
    
    # Check deadline
    if time.time() - _T0 > _DEADLINE:
        return np.clip(result, -100, 100)
    
    # Compute ensemble mean
    ens_mean = np.mean(result, axis=0)
    
    # Compute ensemble covariance
    X_anom = result - ens_mean  # (16, 20)
    B = (X_anom.T @ X_anom) / N
    
    # Localization: apply exponential decay to covariance based on distance
    loc_radius = 4
    loc_mask = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            dist = min(abs(i - j), M - abs(i - j))
            if dist <= loc_radius:
                loc_mask[i, j] = np.exp(-0.5 * (dist / loc_radius) ** 2)
    
    B_loc = B * loc_mask
    
    # Compute Kalman gain
    HBHT = np.zeros((K, K))
    for ki, idx_i in enumerate(observed_indices):
        for kj, idx_j in enumerate(observed_indices):
            HBHT[ki, kj] = B_loc[idx_i, idx_j]
    
    R = observation_variance * np.eye(K)
    S = HBHT + R
    
    BHt = np.zeros((M, K))
    for ki, idx in enumerate(observed_indices):
        BHt[:, ki] = B_loc[:, idx]
    
    try:
        S_inv = np.linalg.inv(S)
        K_gain = BHt @ S_inv
    except np.linalg.LinAlgError:
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        return np.clip(result, -100, 100)
    
    # Innovation
    innovation = observations - ens_mean[observed_indices]
    
    # Update ensemble mean
    ens_mean_new = ens_mean + K_gain @ innovation
    
    # Use stochastic observation perturbation
    rng = np.random.default_rng()
    y_pert = observations + rng.normal(0, np.sqrt(observation_variance), size=(N, K))
    
    # For each member, compute innovation and update
    for i in range(N):
        Hx_i = result[i, observed_indices]
        innov_i = y_pert[i] - Hx_i
        result[i] = result[i] + K_gain @ innov_i
    
    # Adaptive inflation
    spread_obs = np.std(result[:, observed_indices], axis=0)
    mean_spread = np.mean(spread_obs)
    
    if mean_spread < 0.3 * np.sqrt(observation_variance):
        inflation_factor = 1.05
        result = ens_mean_new + inflation_factor * (result - ens_mean_new)
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    return result
