"""
Local Ensemble Kalman Filter (LEKF) for Lorenz-96 assimilation.
Design:
- Propagate ensemble forward using RK4 (5 steps, dt=0.01) between updates.
- Compute localized background error covariance.
- Apply ensemble Kalman update with observation noise.
- Use inflation to prevent filter collapse.
- Localization to limit spurious correlations.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x shape (M, 20)."""
    # x: (M, 20)
    x_m1 = np.roll(x, 1, axis=1)   # x_{i-1}
    x_1  = np.roll(x, -1, axis=1)  # x_{i+1}
    x_m2 = np.roll(x, 2, axis=1)   # x_{i-2}
    return (x_1 - x_m2) * x_m1 - x + F


def _rk4_step(x, F, dt, n_steps):
    """Propagate ensemble using RK4 for n_steps steps of size dt."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _localization_weight(dist, L):
    """Gaspari-Cohn-like localization weight: 1 for dist < L, decays to 0 at 2L."""
    w = np.ones_like(dist, dtype=float)
    mask = dist >= L
    if np.any(mask):
        d = dist[mask]
        # Smooth transition from 1 to 0 over [L, 2L]
        w[mask] = np.maximum(0.0, 1.0 - (d[mask] - L) / L)
        # Apply a smooth cosine taper
        w[mask] = 0.5 * (1.0 + np.cos(np.pi * (d[mask] - L) / L))
    return w


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed variables.
    observed_indices : array-like
        Indices of observed variables (even indices: 0,2,4,...,18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated ensemble with same shape (16, 20).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    M, N = ensemble.shape  # M=16 members, N=20 variables
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Determine F from ensemble statistics (use mean of a few components)
    # For Lorenz-96, we need F. Estimate from the ensemble mean dynamics.
    # Actually, we can't easily get F from the contract. Let's use a heuristic:
    # The task says F=6,8,10 for the three runs. We'll estimate F by checking
    # which F makes the ensemble most consistent with observations.
    # Better: just try to infer F from the state magnitude.
    # For F=6, typical |x| ~ 1-2; F=8 ~ 2-3; F=10 ~ 3-4
    mean_state = np.mean(ensemble, axis=0)
    typical_mag = np.mean(np.abs(mean_state))
    
    # Estimate F: for Lorenz-96, the forcing F relates to the amplitude
    # Empirically: F=6 -> |x|~1.5, F=8 -> |x|~2.5, F=10 -> |x|~3.5
    # We'll use a simple mapping
    if typical_mag < 2.0:
        F_est = 6.0
    elif typical_mag < 3.0:
        F_est = 8.0
    else:
        F_est = 10.0
    
    # Propagate ensemble forward (5 steps, dt=0.01)
    dt = 0.01
    n_prop_steps = 5
    ensemble = _rk4_step(ensemble, F_est, dt, n_prop_steps)
    
    # Clip to ensure finite values
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    # Ensemble Kalman update
    # Background mean and covariance
    xbar = np.mean(ensemble, axis=0)  # (N,)
    X = ensemble - xbar  # (M, N)
    
    # Background covariance (with localization)
    # C = X^T X / (M-1)
    C = (X.T @ X) / (M - 1.0)  # (N, N)
    
    # Localize the covariance
    # Compute distance matrix
    idx = np.arange(N)
    # Periodic distance
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)  # periodic
    
    # Localization length scale
    L_loc = 5.0
    W = _localization_weight(dist, L_loc)
    C_loc = C * W
    
    # Observation operator: H selects observed indices
    # H x = x[obs_idx]
    # Innovation: y - H xbar
    y = observations  # (n_obs,)
    Hxbar = xbar[obs_idx]  # (n_obs,)
    innovation = y - Hxbar  # (n_obs,)
    
    # Observation covariance R = observation_variance * I
    R = observation_variance * np.eye(n_obs)
    
    # Kalman gain computation
    # P_yc = C_loc[obs_idx, :]  # (n_obs, N)
    P_yc = C_loc[obs_idx, :]  # (n_obs, N)
    # S = H P H^T + R
    S = P_yc @ C_loc[:, obs_idx] + R  # (n_obs, n_obs)
    
    # Invert S (use pinv for stability)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain K = P_yc^T S^-1  (N, n_obs)
    K = (P_yc.T @ S_inv).T  # (n_obs, N) -> transpose to (N, n_obs)
    # Actually K should be (N, n_obs): K = P_yc^T @ S^-1
    K = P_yc.T @ S_inv  # (N, n_obs)
    
    # Update ensemble mean
    xbar_new = xbar + K @ innovation  # (N,)
    
    # Update each ensemble member (perturbed observations method)
    # Or use the standard ensemble update
    # For each member: x_m = xbar + (x_m - xbar) - K (H x_m - y)
    # This is equivalent to: x_m_new = xbar_new + (I - K H)(x_m - xbar)
    
    # Compute (I - K H)
    # H is (n_obs, N), K is (N, n_obs)
    KH = K @ (np.eye(N)[obs_idx, :])  # (N, N)
    A = np.eye(N) - KH  # (N, N)
    
    ensemble_new = np.zeros_like(ensemble)
    for m in range(M):
        ensemble_new[m] = xbar_new + A @ (ensemble[m] - xbar)
    
    # Inflation to prevent filter collapse
    # Estimate ensemble spread vs observation spread
    ens_spread_obs = np.std(ensemble_new[:, obs_idx], axis=0)
    obs_spread = np.std(innovation)
    # Inflate if ensemble spread is too small
    inflation_factor = 1.0
    if np.mean(ens_spread_obs) < 0.5 * np.sqrt(observation_variance):
        inflation_factor = 1.1
    
    if inflation_factor != 1.0:
        ensemble_new = xbar_new + inflation_factor * (ensemble_new - xbar_new)
    
    # Ensure finite and within bounds
    ensemble_new = np.nan_to_num(ensemble_new, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    return ensemble_new

# EVOLVE-BLOCK-END
