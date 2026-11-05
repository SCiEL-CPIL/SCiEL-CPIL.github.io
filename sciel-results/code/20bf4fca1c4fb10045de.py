"""Lorenz-96 assimilation using a localized ensemble Kalman filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_step(x, F, dt=0.01):
    """
    One RK4 step for the Lorenz-96 system.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    with periodic boundary conditions.
    """
    n = len(x)
    
    def f(x):
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        return dx
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized ensemble Kalman filter analysis update with adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Compute ensemble covariance
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Get observed state components from ensemble
    ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
    ens_obs_mean = np.mean(ens_obs, axis=0)
    
    # Compute innovation
    innovation = observations - ens_obs_mean
    
    # Compute innovation covariance S = H P H^T + R
    # H P H^T is just the submatrix of cov for observed indices
    P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
    S = P_obs_obs + R
    
    # Regularize S
    S_reg = S + 1e-8 * np.eye(n_obs)
    
    # Compute Kalman gain K = P H^T S^{-1}
    # P H^T is the covariance between all states and observed states
    PHt = cov[:, observed_indices]  # (n_state, n_obs)
    
    try:
        K = np.linalg.solve(S_reg, PHt.T).T
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S_reg)
    
    # Apply localization to the Kalman gain using periodic distance
    decay_length = 3.0
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    
    # Update each ensemble member
    member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
    update = member_innovations @ localized_K.T  # (n_members, n_state)
    ens = ens + update
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Expected innovation variance after update
    P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_new + R
    
    # Compute component-wise inflation factors
    inflation_factors = np.ones(n_state)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        exp_var = S_expected[j, j]
        if exp_var > 1e-10:
            ratio = (innovation[j] ** 2 + observation_variance) / exp_var
            # Smooth the ratio to avoid extreme values
            ratio = np.clip(ratio, 0.5, 2.0)
            # Apply to observed component and nearby unobserved components
            for i in range(n_state):
                d = abs(i - obs_idx)
                d = min(d, n_state - d)
                w = np.exp(-d ** 2 / (2.0 * 2.0 ** 2))
                inflation_factors[i] *= (1.0 + 0.5 * w * (np.sqrt(ratio) - 1.0))
    
    # Global inflation based on overall spread
    global_ratio = np.mean([
        (innovation[j] ** 2 + observation_variance) / S_expected[j, j]
        for j in range(n_obs) if S_expected[j, j] > 1e-10
    ])
    global_inflation = np.sqrt(np.clip(global_ratio, 0.9, 1.3))
    
    # Combine component-wise and global inflation
    combined_inflation = 0.7 * inflation_factors + 0.3 * global_inflation
    combined_inflation = np.clip(combined_inflation, 0.85, 1.35)
    
    # Apply inflation around the new mean
    ens = ens_mean_new + combined_inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
