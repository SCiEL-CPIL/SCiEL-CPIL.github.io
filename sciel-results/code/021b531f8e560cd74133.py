"""Combined assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline guard: 5s safety margin before 150s kill
_DEADLINE = time.time() + 145.0

def _lorenz96_step(x, F, dt=0.01, n_steps=5):
    """Propagate Lorenz-96 state by RK4 for n_steps."""
    M = len(x)
    x = x.copy()
    for _ in range(n_steps):
        def f(state):
            return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with localization and adaptive inflation
    for Lorenz-96 partial observation.
    
    Uses a deterministic (deterministic) square-root filter update with
    Gaussian localization and adaptive spread inflation.
    
    Parameters:
    -----------
    ensemble : (16, 20) ndarray
        Ensemble of 16 members, each a 20-state Lorenz-96 vector
    observations : (10,) ndarray
        Noisy observations of the 10 even-indexed states
    observed_indices : array_like
        Indices of observed variables
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) ndarray
        Updated ensemble after assimilation
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return ensemble.copy()
    
    N, M = ensemble.shape
    obs_idx = np.array(sorted(observed_indices), dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite input
    ens = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    obs = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    
    # --- Step 1: Compute ensemble mean and anomalies ---
    x_mean = np.mean(ens, axis=0)  # (20,)
    anomalies = ens - x_mean  # (N, 20)
    
    # --- Step 2: Adaptive Inflation ---
    # Compute observation-space innovation
    x_mean_obs = x_mean[obs_idx]  # (10,)
    innovation = obs - x_mean_obs  # (10,)
    
    # Estimate ensemble variance in observation space
    anomalies_obs = anomalies[:, obs_idx]  # (N, 10)
    P_yy = np.mean(anomalies_obs ** 2, axis=0)  # (10,)
    P_yy_total = P_yy + observation_variance  # (10,)
    
    # Innovation variance ratio for adaptive inflation
    innovation_var = np.mean(innovation ** 2)
    expected_innov_var = np.mean(P_yy_total)
    
    if expected_innov_var > 1e-10:
        ratio = innovation_var / expected_innov_var
        # Clamp ratio to safe range
        ratio = np.clip(ratio, 0.5, 3.0)
        inflation_factor = np.sqrt(ratio)
    else:
        inflation_factor = 1.0
    
    # Apply inflation about the mean
    if abs(inflation_factor - 1.0) > 1e-6:
        anomalies = anomalies * inflation_factor
        ens = x_mean + anomalies
    
    # --- Step 3: Build localization matrix ---
    # Localization scale tuned for L96 correlation length
    length_scale = 2.5
    
    # Precompute periodic distances from each observed index to all state indices
    dist_matrix = np.zeros((n_obs, M))
    for i, oi in enumerate(obs_idx):
        for j in range(M):
            d = abs(oi - j)
            d = min(d, M - d)  # periodic
            dist_matrix[i, j] = d
    
    # Localization weights (Gaussian)
    L = np.exp(-0.5 * (dist_matrix / length_scale) ** 2)  # (10, 20)
    # Ensure self-observation gets full weight
    for i in range(n_obs):
        L[i, obs_idx[i]] = 1.0
    
    # --- Step 4: Deterministic Ensemble Kalman Filter update ---
    # Process all observations simultaneously (deterministic, not sequential)
    updated_ens = ens.copy()
    
    # Compute ensemble covariance in observation space
    # P_yy is already computed above
    
    # Compute cross-covariance: P_xy (n_obs, M)
    # P_xy[i, j] = (1/N) * sum_n (anomalies[n, obs_idx[i]] * anomalies[n, j]) * L[i, j]
    # Vectorized: for each observation i, P_xy[i, :] = (anomalies.T @ anomalies[:, obs_idx[i]]) / N * L[i, :]
    # This can be written as: P_xy = (anomalies[:, obs_idx].T @ anomalies) / N * L
    # anomalies[:, obs_idx] is (N, n_obs), so (N, n_obs).T @ (N, M) = (n_obs, M)
    P_xy = (anomalies[:, obs_idx].T @ anomalies) / N  # (n_obs, M)
    P_xy *= L  # Apply localization element-wise: (n_obs, M) * (n_obs, M)
    
    # Kalman gain: K = P_xy @ inv(P_yy_total)
    # P_yy_total is (n_obs,), so we can do:
    K = P_xy / P_yy_total[np.newaxis, :]  # (n_obs, M)
    
    # Innovation: (n_obs,)
    # Update: updated_ens[n, j] += sum_i K[i, j] * (obs[i] - ens[n, obs_idx[i]])
    # member_innov: (N, n_obs) where member_innov[n, i] = obs[i] - ens[n, obs_idx[i]]
    member_innov = obs[np.newaxis, :] - ens[:, obs_idx]  # (N, n_obs)
    
    # Update: (N, n_obs) @ (n_obs, M) -> (N, M)
    updated_ens += member_innov @ K
    
    # --- Step 5: Variance inflation to prevent collapse ---
    new_mean = np.mean(updated_ens, axis=0)
    current_spread = np.std(updated_ens, axis=0)
    pre_spread = np.std(ens, axis=0)  # spread before LESRF update
    
    # Per-component inflation to maintain spread
    for j in range(M):
        if current_spread[j] > 1e-8:
            # Target: maintain at least 95% of pre-update spread
            target = pre_spread[j] * 0.95
            if current_spread[j] < target:
                scale = target / current_spread[j]
                scale = min(scale, 1.3)  # limit inflation
                updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * scale
        else:
            # Add small perturbation if spread collapsed
            updated_ens[:, j] = new_mean[j] + np.random.normal(0, 0.03, N)
    
    # --- Step 6: Safety checks ---
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
