"""
Local Ensemble Kalman Filter (LEKF) for Lorenz-96 assimilation.
Uses ensemble-based covariance estimation with localization and inflation.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20)."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def _localize_cov(cov, decay_length=2.5):
    """Apply exponential localization to covariance matrix (20x20)."""
    n = cov.shape[0]
    loc = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            dist = min(abs(i - j), n - abs(i - j))
            loc[i, j] = np.exp(-0.5 * (dist / decay_length) ** 2)
    return cov * loc

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter analysis update.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (16, 10) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar observation noise variance
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    N = ensemble.shape[0]  # number of ensemble members
    n = ensemble.shape[1]  # number of state variables
    n_obs = len(observed_indices)
    
    # Step 1: Propagate ensemble forward 5 RK4 steps with dt=0.01
    dt = 0.01
    n_steps = 5
    
    # Try different F values - we need to infer F from the ensemble
    # Since we don't have F directly, we'll estimate it or use a reasonable default
    # The task says F is 6, 8, or 10. We'll try to estimate from the ensemble.
    # For now, use a simple approach: try each F and pick the one that keeps ensemble bounded
    
    # Actually, let's just use the dynamics with an estimated F
    # We can estimate F by looking at the equilibrium: for Lorenz-96, the fixed point is x_i = F
    # So the mean of the ensemble should be close to F
    F_est = np.mean(ensemble)
    
    # Round to nearest integer for stability
    F_candidates = [6.0, 8.0, 10.0]
    F = min(F_candidates, key=lambda f: abs(f - F_est))
    
    # Propagate
    ensemble_prop = _rk4_step(ensemble.copy(), F, dt, n_steps)
    
    # Step 2: Compute ensemble mean and covariance
    mean = np.mean(ensemble_prop, axis=0)
    anomalies = ensemble_prop - mean
    
    # Background covariance (20x20)
    P = np.cov(anomalies, rowvar=False, bias=True)
    
    # Localize the covariance
    P = _localize_cov(P, decay_length=2.5)
    
    # Step 3: Observation operator H (selects observed indices)
    H = np.zeros((n_obs, n))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation noise covariance R
    R = observation_variance * np.eye(n_obs)
    
    # Step 4: Compute Kalman gain
    # S = H P H^T + R
    HP = H @ P
    S = HP @ H.T + R
    
    # K = P H^T S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P @ H.T @ S_inv  # (20, n_obs)
    
    # Step 5: Compute innovation (observation minus predicted observation)
    # Predicted observations: H @ mean
    y_pred = H @ mean
    innovation = observations.T - y_pred  # (n_obs, N) if observations is (N, n_obs)
    
    # Wait, observations shape is (16, 10) = (N, n_obs)
    # So innovation should be (n_obs, N) or we work with (N, n_obs)
    
    # Let's work with the ensemble directly
    # For each ensemble member, compute the update
    updated_ensemble = ensemble_prop.copy()
    
    for m in range(N):
        # Innovation for this member
        y_m = H @ ensemble_prop[m]  # (n_obs,)
        innov_m = observations[m] - y_m  # (n_obs,)
        
        # Update
        updated_ensemble[m] = ensemble_prop[m] + K @ innov_m
    
    # Step 6: Ensemble inflation to prevent collapse
    # Compute ensemble spread
    spread = np.std(updated_ensemble, axis=0)
    target_spread = np.std(ensemble, axis=0)  # original spread
    
    # Inflate if spread is too small
    for i in range(n):
        if spread[i] < 0.5 * target_spread[i]:
            factor = target_spread[i] / max(spread[i], 1e-10)
            factor = min(factor, 1.5)  # cap inflation
            updated_ensemble[:, i] = mean[i] + factor * (updated_ensemble[:, i] - mean[i])
    
    # Clip to reasonable bounds
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
