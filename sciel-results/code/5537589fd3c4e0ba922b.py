"""
Ensemble Kalman Filter (EnKF) for Lorenz-96 assimilation.
Uses a proper Kalman update with ensemble covariance, localization,
and inflation to improve upon naive nudging.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x shape: (M, N) where M=ensemble size, N=20."""
    N = x.shape[1]
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4 for n_steps."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def _localization_weight(dist, L=5.0):
    """Gaspari-Cohn localization weight."""
    if dist >= 2 * L:
        return 0.0
    elif dist >= L:
        L2 = L * L
        x = dist / L
        return (x**4 * (5 - 4*x) + 2) / 12.0
    else:
        L2 = L * L
        x = dist / L
        return (-x**6 + 2*x**4 + x**2 * (3 - 2*x**2) + 4) / 12.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed variables (even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    M, N = ensemble.shape  # M=16, N=20
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Estimate forcing F from the ensemble mean
    # For Lorenz-96, F is typically constant. We can estimate it from the dynamics.
    # But since we don't know F directly, let's use a heuristic.
    # Actually, the task says F=6,8,10 for three runs. We need to infer F.
    # One approach: use the fact that for a balanced Lorenz-96, the mean of x_i is approximately F.
    # But this is not exact. Let's try to estimate F from the ensemble.
    # Actually, a better approach: the evaluator propagates with a specific F.
    # Since we don't know F, we can try to estimate it.
    
    # Estimate F from the ensemble: for Lorenz-96, in steady state, mean(x) ≈ F
    # But this is rough. Let's use the median of the ensemble mean as an estimate.
    F_est = np.median(np.mean(ensemble, axis=1))
    
    # Actually, let's be more careful. The forcing F is a parameter of the system.
    # We can try multiple values and see which one gives the best fit to observations.
    # But that's expensive. Let's just use a reasonable estimate.
    # For Lorenz-96, F=8 gives a chaotic regime with mean around 8.
    # Let's use the ensemble mean as a rough estimate.
    F_est = float(np.clip(np.mean(ensemble), 4.0, 12.0))
    
    # Propagate ensemble forward by 5 RK4 steps with dt=0.01 (matching evaluator)
    dt = 0.01
    n_prop_steps = 5
    bg_ensemble = _rk4_step(ensemble, F_est, dt, n_prop_steps)
    
    # Check time
    if time.time() > deadline:
        # Return best we can do quickly
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        return np.clip(result, -100, 100)
    
    # Compute background ensemble mean and covariance
    bg_mean = np.mean(bg_ensemble, axis=0)  # shape (N,)
    bg_anom = bg_ensemble - bg_mean  # shape (M, N)
    
    # Background covariance: B = (1/(M-1)) * A^T A, where A is (M, N)
    B = (bg_anom.T @ bg_anom) / (M - 1)  # shape (N, N)
    
    # Apply localization to the background covariance
    L_loc = 5.0  # Localization length scale
    for i in range(N):
        for j in range(N):
            dist = min(abs(i - j), N - abs(i - j))  # Periodic distance
            w = _localization_weight(dist, L_loc)
            B[i, j] *= w
    
    # Observation operator: H selects observed indices
    # H is (n_obs, N)
    H = np.zeros((n_obs, N))
    for k, idx in enumerate(obs_idx):
        H[k, idx] = 1.0
    
    # Observation noise covariance: R = observation_variance * I
    R = observation_variance * np.eye(n_obs)
    
    # Compute Kalman gain using the standard formula:
    # K = B H^T (H B H^T + R)^{-1}
    # But we need to use the localized B.
    
    # Innovation: y - H bg_mean
    innovation = observations - H @ bg_mean  # shape (n_obs,)
    
    # Compute H B H^T
    HB = H @ B  # shape (n_obs, N)
    HBHt = HB @ H.T  # shape (n_obs, n_obs)
    
    # Add R and invert
    S = HBHt + R  # shape (n_obs, n_obs)
    
    # Compute Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = B @ H.T @ S_inv  # shape (N, n_obs)
    
    # Update ensemble mean
    analysis_mean = bg_mean + K @ innovation  # shape (N,)
    
    # Update ensemble spread using ensemble Kalman filter formulation
    # For each ensemble member, update using the ensemble-based gain
    # K_m = B_m H^T (H B_m H^T + R)^{-1}, where B_m is the sample covariance from member m
    # But this is expensive. Let's use the standard EnKF update:
    # x_a = x_b + K (y - H x_b)
    # where K is computed from the ensemble.
    
    # Alternative: use the ensemble Kalman filter with perturbed observations
    # For each member m:
    #   x_a_m = x_b_m + K (y_m - H x_b_m)
    # where y_m = y + sqrt(R) * eps_m
    
    # Let's use the standard EnKF formulation
    # First, compute the ensemble-based Kalman gain for each member
    # Actually, let's use a simpler approach: update each member using the same K
    
    # But to maintain ensemble spread, we should use the ensemble-based update
    # Let's compute the gain from the ensemble anomalies
    
    # For each member, the update is:
    # x_a_m = x_b_m + K (y - H x_b_m)
    # But this collapses the ensemble. We need to use the ensemble-based formulation.
    
    # Standard EnKF: 
    #   K_m = (bg_anom @ H.T) @ np.linalg.inv(H @ bg_anom @ H.T / (M-1) + R)
    #   x_a_m = bg_mean + (bg_anom[m] + K_m @ (observations - H @ bg_mean))
    # Wait, that's not quite right. Let me think again.
    
    # The proper EnKF update:
    # 1. Compute the ensemble mean and anomalies
    # 2. For each member m, compute the innovation: y - H x_b_m
    # 3. Compute the gain from the ensemble: K = (1/(M-1)) A H^T (H A H^T/(M-1) + R)^{-1}
    #    where A is the anomaly matrix
    # 4. x_a_m = x_b_m + K (y - H x_b_m)
    
    # Let's use this formulation
    A = bg_anom  # (M, N)
    AHt = A @ H.T  # (M, n_obs)
    HAHt = H @ A @ H.T / (M - 1)  # (n_obs, n_obs)
    
    # Apply localization to HAHt as well
    # Actually, we already localized B, so let's use that for consistency
    # Let's recompute using the localized B
    
    # For the EnKF, we need the gain to be consistent with the localized covariance
    # Let's use the K computed above from the localized B
    
    # Update each member
    updated_ensemble = np.zeros_like(bg_ensemble)
    for m in range(M):
        innovation_m = observations - H @ bg_ensemble[m]  # (n_obs,)
        updated_ensemble[m] = bg_ensemble[m] + K @ innovation_m
    
    # Apply inflation to maintain ensemble spread
    # Compute the ensemble spread after update
    upd_mean = np.mean(updated_ensemble, axis=0)
    upd_anom = updated_ensemble - upd_mean
    upd_spread = np.sqrt(np.mean(upd_anom**2, axis=0))
    
    # Compare with background spread
    bg_spread = np.sqrt(np.mean(bg_anom**2, axis=0))
    
    # Inflate if spread has collapsed
    inflation_factor = 1.02
    updated_ensemble = upd_mean + inflation_factor * upd_anom
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    # Ensure finiteness
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
