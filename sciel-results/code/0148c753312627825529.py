"""Lorenz-96 ensemble assimilation: Deterministic EnKF with adaptive multiplicative inflation and Gaussian localization."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.05
_inflation_alpha = 0.85
_inflation_min = 1.0
_inflation_max = 2.5
_loc_range = 2.8

# Precompute localization weights
_loc_cache = {}

def _get_loc_weights(obs_idx, dim=20, rng=2.8):
    key = (obs_idx, dim, rng)
    if key not in _loc_cache:
        w = np.empty(dim)
        for j in range(dim):
            d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            w[j] = np.exp(-0.5 * (d / rng) ** 2)
        _loc_cache[key] = w
    return _loc_cache[key]

def _lorenz96_deriv(x, F=8.0):
    """Compute Lorenz-96 derivatives. x has shape (N, dim)."""
    N, dim = x.shape
    dx = np.empty_like(x)
    # Periodic indices: i-2, i-1, i, i+1 mod dim
    xm2 = np.roll(x, 2, axis=1)
    xm1 = np.roll(x, 1, axis=1)
    xp1 = np.roll(x, -1, axis=1)
    dx = (xm2 - xm1) * xp1 - x + F
    return dx

def _rk4_step(x, dt, F=8.0):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_deriv(x, F)
    k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_deriv(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # Observation operator
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # --- 1. Compute Mean and Anomalies ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (N, dim)
    
    # --- 2. Adaptive Inflation ---
    y_f_mean = H @ mean
    innovation = observations - y_f_mean  # (n_obs,)
    
    # Covariance of forecast ensemble
    cov = np.cov(ensemble, rowvar=False)
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation covariance S = H P H^T + R
    P_obs = cov[np.ix_(observed_indices, observed_indices)]
    S = P_obs + R
    S += 1e-10 * np.eye(n_obs)
    
    # Calculate innovation norm and expected norm
    innovation_norm_sq = innovation @ innovation
    expected_norm_sq = np.trace(S)
    
    # Adaptive inflation update
    if expected_norm_sq > 1e-15:
        ratio = np.sqrt(innovation_norm_sq / expected_norm_sq)
        target_inflation = max(1.0, min(ratio, _inflation_max))
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target_inflation
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # --- 3. Inflate Covariance ---
    P_inf = (_inflation ** 2) * cov
    
    # --- 4. Compute Kalman Gain ---
    P_obs_inf = P_inf[np.ix_(observed_indices, observed_indices)]
    S_inf = P_obs_inf + R
    S_inf += 1e-10 * np.eye(n_obs)
    
    PHt = P_inf @ H.T
    try:
        S_inv = np.linalg.inv(S_inf)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_inf)
    
    K = PHt @ S_inv  # (dim, n_obs)
    
    # Apply Localization to Gain
    K_loc = K.copy()
    for j, obs_idx in enumerate(observed_indices):
        w = _get_loc_weights(obs_idx, dim, _loc_range)
        K_loc[:, j] *= w
    
    # --- 5. Update Mean ---
    mean_update = K_loc @ innovation
    mean_a = mean + mean_update
    
    # --- 6. Update Anomalies (Deterministic EnKF) ---
    KH = K_loc @ H
    anomalies_a_T = anomalies.T - KH @ anomalies.T
    anomalies_a = anomalies_a_T.T
    
    # --- 7. Post-Inflation ---
    anomalies_a = _inflation * anomalies_a
    
    # --- 8. Final Ensemble ---
    updated = mean_a + anomalies_a
    
    # --- 9. ES-4DVar Nudging: Iteratively refine using dynamics ---
    # We perform a few iterations of nudging toward observations while respecting dynamics
    # This is a simplified ES-4DVar approach: propagate forward, compare to obs, adjust
    
    # Estimate F from the system (use 8.0 as default, or estimate from data)
    # For now, use F=8.0 as a reasonable middle ground
    F_est = 8.0
    
    # Number of nudging iterations (keep small for speed)
    n_iter = 3
    dt = 0.01
    n_steps = 5
    
    # Current best ensemble
    best_ensemble = updated.copy()
    best_cost = np.inf
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Propagate ensemble forward by n_steps * dt
        ens_prop = best_ensemble.copy()
        for step in range(n_steps):
            ens_prop = _rk4_step(ens_prop, dt, F_est)
        
        # Compute cost: sum of squared innovations at observation points
        ens_mean_prop = ens_prop.mean(axis=0)
        obs_pred = ens_mean_prop[observed_indices]
        cost = np.sum((obs_pred - observations) ** 2)
        
        if cost < best_cost:
            best_cost = cost
            best_ensemble = ens_prop.copy()
        
        # Compute gradient-based adjustment
        # J = sum of (H x_prop - y)^2
        # We nudge the initial condition to reduce cost
        # Simple approach: compute sensitivity and adjust
        
        # For simplicity, use a relaxation approach
        # Compute innovation in the propagated state
        innovation_prop = observations - obs_pred
        
        # Back-propagate adjustment using adjoint-like approximation
        # This is a simplification: we assume the linearized dynamics
        # and use the Kalman gain to map innovation back to initial state
        
        # Use the gain to compute adjustment to the initial ensemble
        # K maps observation innovation to state adjustment
        # We apply a fraction of the adjustment
        alpha = 0.5  # Relaxation factor
        
        # Adjust the initial ensemble
        adjustment = K_loc @ innovation_prop * alpha
        new_ensemble = best_ensemble + np.tile(adjustment, (N, 1))
        
        # Check if this improves things
        ens_test = new_ensemble.copy()
        for step in range(n_steps):
            ens_test = _rk4_step(ens_test, dt, F_est)
        
        ens_mean_test = ens_test.mean(axis=0)
        obs_pred_test = ens_mean_test[observed_indices]
        cost_test = np.sum((obs_pred_test - observations) ** 2)
        
        if cost_test < best_cost:
            best_cost = cost_test
            best_ensemble = new_ensemble.copy()
        # else keep best_ensemble
    
    updated = best_ensemble
    
    # Safety: time guard
    if time.time() - t0 > 100:
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)
    
    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
