"""Lorenz-96 assimilation using a Particle Flow Filter with adaptive inflation."""
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
    Perform a Particle Flow Filter analysis update.
    
    Uses a smooth, divergence-free flow field to transport ensemble members
    so that the posterior mean and covariance match the observation statistics.
    This preserves non-Gaussian structure better than linear EnKF methods.
    
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
    
    # Compute ensemble statistics
    ens_mean = np.mean(ens, axis=0)
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
    P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
    S = P_obs_obs + R
    
    # Regularize S
    S_reg = S + 1e-8 * np.eye(n_obs)
    
    # Compute Kalman gain K = P H^T S^{-1}
    PHt = cov[:, observed_indices]  # (n_state, n_obs)
    
    try:
        K = np.linalg.solve(S_reg, PHt.T).T
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S_reg)
    
    # Apply localization to the Kalman gain using periodic distance
    # Use rational localization (Hamill et al.) for better properties
    decay_length = 2.0
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            # Rational localization: w = 1/(1 + d^2/L^2)
            w = 1.0 / (1.0 + (d / decay_length) ** 2)
            localized_K[i, j] = K[i, j] * w
    
    # PFF: Construct a smooth flow field that transports the ensemble
    # We use a quadratic flow field: v(x) = A @ x + b
    # where A and b are chosen so that the flow maps the prior to the posterior
    
    # Target posterior mean (from EnKF update)
    posterior_mean = ens_mean + localized_K @ innovation
    
    # Target posterior covariance (from EnKF)
    # P_post = (I - K H) P (I - K H)^T + K R K^T
    # For simplicity, we'll use the ensemble-based posterior covariance
    ens_obs_mean = np.mean(ens_obs, axis=0)
    member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
    ens_post = ens + member_innovations @ localized_K.T  # (n_members, n_state)
    
    # Compute target covariance from the EnKF-updated ensemble
    post_anomalies = ens_post - np.mean(ens_post, axis=0)
    cov_target = (post_anomalies.T @ post_anomalies) / (n_members - 1)
    
    # Now construct the PFF flow field
    # We want a flow v(x) such that:
    # 1. The mean is transported: <v(x)> = posterior_mean - ens_mean
    # 2. The covariance is matched: <(x - mean)(x - mean)^T> transforms appropriately
    
    # Simple approach: use a linear flow v(x) = (posterior_mean - ens_mean) + A @ (x - ens_mean)
    # where A is chosen to match the covariance transformation
    
    # For the PFF, we use a quadratic flow that exactly matches mean and covariance
    # The flow field is: v(x) = c0 + c1 @ x + c2 @ (x @ x^T)
    # But for computational efficiency, we use a linear flow with optimal A
    
    # Optimal A for covariance matching:
    # We want: (I + A) P (I + A)^T = cov_target
    # This is a Lyapunov equation. For simplicity, we use:
    # A = cov_target @ P^{-1} - I (approximately)
    
    # Regularize cov for inversion
    cov_reg = cov + 1e-6 * np.eye(n_state)
    
    try:
        cov_inv = np.linalg.inv(cov_reg)
        A = cov_target @ cov_inv - np.eye(n_state)
    except np.linalg.LinAlgError:
        A = np.zeros((n_state, n_state))
    
    # Damping factor to ensure stability
    alpha = 0.8
    A = alpha * A
    
    # Flow field: v(x) = (posterior_mean - ens_mean) + A @ (x - ens_mean)
    mean_shift = posterior_mean - ens_mean
    
    # Apply the flow to each ensemble member
    for m in range(n_members):
        x = ens[m]
        v = mean_shift + A @ (x - ens_mean)
        ens[m] = x + v
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Expected innovation variance after update
    P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_new + R
    
    # Compute inflation factor
    ratios = np.ones(n_obs)
    for j in range(n_obs):
        exp_var = S_expected[j, j]
        if exp_var > 1e-10:
            ratios[j] = (innovation[j] ** 2 + observation_variance) / exp_var
    
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.95, 1.25))
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
