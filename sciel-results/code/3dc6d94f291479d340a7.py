"""Lorenz-96 ensemble Kalman filter with localization and adaptive inflation."""
import numpy as np

def _lorenz96_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x)
    def f(state):
        dx = np.empty_like(state)
        for i in range(n):
            dx[i] = (state[(i+1) % n] - state[(i-2) % n]) * state[(i-1) % n] - state[i] + F
        return dx
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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
    
    # Compute ensemble mean
    ens_mean = np.mean(result, axis=0)
    
    # Compute ensemble covariance (16 x 20)
    X_anom = result - ens_mean  # (16, 20)
    
    # Compute background covariance B = (1/N) * X_anom^T @ X_anom
    # But for efficiency, work with the ensemble directly
    # B = X_anom.T @ X_anom / N  (20 x 20)
    B = (X_anom.T @ X_anom) / N
    
    # Observation operator H: selects observed indices
    # H @ x = x[observed_indices]
    # H is (K, M) sparse, but we can work implicitly
    
    # Innovation: y - H @ x_mean
    Hx_mean = ens_mean[observed_indices]
    innovation = observations - Hx_mean  # (K,)
    
    # Observation error covariance R = observation_variance * I (K x K)
    R = observation_variance * np.eye(K)
    
    # Localization: apply exponential decay to covariance based on distance
    # For Lorenz-96, correlations decay with distance; use radius of influence
    # Localize B by zeroing out far-field correlations
    loc_radius = 4  # only keep correlations within 4 indices (circular)
    loc_mask = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            dist = min(abs(i - j), M - abs(i - j))
            if dist <= loc_radius:
                # Exponential localization
                loc_mask[i, j] = np.exp(-0.5 * (dist / loc_radius) ** 2)
    
    B_loc = B * loc_mask
    
    # Compute Kalman gain: K = B_loc @ H^T @ (H @ B_loc @ H^T + R)^{-1}
    # H @ B_loc @ H^T is (K x K)
    HBHT = np.zeros((K, K))
    for ki, idx_i in enumerate(observed_indices):
        for kj, idx_j in enumerate(observed_indices):
            HBHT[ki, kj] = B_loc[idx_i, idx_j]
    
    # S = HBHT + R (K x K)
    S = HBHT + R
    
    # Solve S @ gain = B_loc @ H^T
    # B_loc @ H^T is (M x K): column j is B_loc[:, observed_indices[j]]
    BHt = np.zeros((M, K))
    for ki, idx in enumerate(observed_indices):
        BHt[:, ki] = B_loc[:, idx]
    
    # Kalman gain K_gain = BHt @ S^{-1}  (M x K)
    try:
        S_inv = np.linalg.inv(S)
        K_gain = BHt @ S_inv
    except np.linalg.LinAlgError:
        # Fallback to simple nudging if singular
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        return result
    
    # Update ensemble mean
    ens_mean_new = ens_mean + K_gain @ innovation
    
    # Update ensemble: x_i' = x_i + K_gain @ (y_i - H @ x_i)
    # where y_i are perturbed observations
    # Use stochastic observation perturbation (SE-Kalman)
    # Perturb observations with N(0, R)
    rng = np.random.default_rng()
    y_pert = observations + rng.normal(0, np.sqrt(observation_variance), size=(N, K))
    
    # For each member, compute innovation and update
    for i in range(N):
        Hx_i = result[i, observed_indices]
        innov_i = y_pert[i] - Hx_i  # (K,)
        result[i] = result[i] + K_gain @ innov_i
    
    # Adaptive inflation: check if ensemble spread is too small
    # Compute spread at observed locations
    spread_obs = np.std(result[:, observed_indices], axis=0)
    mean_spread = np.mean(spread_obs)
    
    # If spread is too small relative to observation error, inflate
    if mean_spread < 0.3 * np.sqrt(observation_variance):
        # Inflate around the mean
        inflation_factor = 1.05
        result = ens_mean_new + inflation_factor * (result - ens_mean_new)
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    return result
