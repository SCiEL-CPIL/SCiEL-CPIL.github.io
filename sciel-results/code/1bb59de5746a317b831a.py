"""Lorenz-96 assimilation using an Ensemble Square Root Filter (EnSRF) with adaptive inflation."""
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
    Perform an Ensemble Square Root Filter (EnSRF) analysis update.
    Uses sequential coordinate-wise regression to update unobserved variables
    based on observed ones, avoiding explicit Kalman gain matrix inversion.
    
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
    
    # Apply localization to the Kalman gain using rational (Hamill et al.) localization
    # This has better properties for sparse ensembles than Gaussian localization
    decay_length = 3.0
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            # Rational localization: w = 1/(1 + (d/L)^2)
            w = 1.0 / (1.0 + (d / decay_length) ** 2)
            localized_K[i, j] = K[i, j] * w
    
    # Apply relaxation to prevent over-correction with small ensembles
    relaxation = 0.95
    localized_K *= relaxation
    
    # EnSRF: Sequential coordinate-wise regression
    # We update the ensemble mean and anomalies separately.
    # The analysis mean is: x_a = x_f + K (y - H x_f)
    # The analysis anomalies are updated via regression.
    
    # First, compute the updated mean
    update_mean = innovation @ localized_K.T  # (n_state,)
    ens_mean_new = ens_mean + update_mean
    
    # For the anomalies, we use the regression approach:
    # For each unobserved state i, we regress the anomaly on the observed anomalies.
    # The regression coefficient for state i given observations j is:
    # beta_ij = Cov(x_i, y_j) / Var(y_j)
    # But since we have multiple observations, we use the full gain structure.
    
    # The analysis anomaly for member m at state i is:
    # a_a_i = a_f_i - sum_j K_ij * (a_f_obs_j - a_obs_j)
    # where a_obs_j is the observation anomaly (but observations are fixed, not ensemble)
    
    # Actually, in EnSRF, we can compute the analysis anomalies directly:
    # The key insight is that the analysis anomalies are a linear transformation of the forecast anomalies.
    
    # Let's use the standard EnSRF approach:
    # 1. Compute the analysis mean
    # 2. For each state, compute the regression of the state on the observed states
    # 3. Update the anomalies accordingly
    
    # Compute the regression coefficients for each state
    # For state i, we want to predict a_i from the observed anomalies
    # beta_i = Cov(x_i, x_obs) @ Var(x_obs)^{-1}
    
    # But we need to account for the observation noise in the regression.
    # The proper EnSRF update for anomalies is:
    # a_a = a_f - K @ (a_f_obs - y_anom)
    # where y_anom is the innovation (but this is for the mean)
    
    # Actually, let's use a cleaner EnSRF formulation:
    # The analysis ensemble is:
    # x_a = x_f + K (y - H x_f) + (I - K H) (x_f - x_f_mean)
    #      = x_f_mean + K (y - H x_f_mean) + (I - K H) (x_f - x_f_mean)
    
    # So the analysis anomalies are: (I - K H) @ forecast_anomalies
    # where H is the observation operator (selects observed indices)
    
    # Build the observation matrix H (sparse)
    # H @ x = x[observed_indices]
    # So (I - K H) @ a_f = a_f - K @ a_f_obs
    
    # a_f is the forecast anomalies: (n_members, n_state)
    # a_f_obs = a_f[:, observed_indices]: (n_members, n_obs)
    
    a_f = ens - ens_mean  # (n_members, n_state)
    a_f_obs = a_f[:, observed_indices]  # (n_members, n_obs)
    
    # Analysis anomalies: a_a = a_f - localized_K @ a_f_obs
    # localized_K is (n_state, n_obs)
    # a_f_obs is (n_members, n_obs)
    # So a_f_obs.T @ localized_K.T is (n_state, n_members) -> transpose to (n_members, n_state)
    
    a_a = a_f - (a_f_obs @ localized_K)  # (n_members, n_state)
    
    # Wait, let's check dimensions:
    # a_f: (n_members, n_state)
    # a_f_obs: (n_members, n_obs)
    # localized_K: (n_state, n_obs)
    # a_f_obs @ localized_K: (n_members, n_obs) @ (n_state, n_obs) -> this doesn't work
    
    # We need: a_f - localized_K.T @ a_f_obs
    # localized_K.T: (n_obs, n_state)
    # a_f_obs: (n_members, n_obs)
    # We need to compute: for each member, a_f[m] - localized_K.T @ a_f_obs[m]
    # = a_f - (a_f_obs @ localized_K)  ... no
    
    # Actually: (I - K H) @ a_f where H selects observed components
    # H @ a_f = a_f_obs (n_members, n_obs)
    # K @ H @ a_f = K @ a_f_obs
    # K is (n_state, n_obs), a_f_obs is (n_members, n_obs)
    # So we need: a_f_obs.T @ K.T -> (n_obs, n_members) @ (n_obs, n_state) -> doesn't work
    
    # Let's think again. For each member m:
    # a_a[m] = a_f[m] - K @ a_f_obs[m]
    # K @ a_f_obs[m] is (n_state,) 
    # So we need: a_f - (a_f_obs @ K.T)
    # a_f_obs: (n_members, n_obs)
    # K.T: (n_obs, n_state)
    # a_f_obs @ K.T: (n_members, n_state) ✓
    
    a_a = a_f - (a_f_obs @ localized_K.T)  # (n_members, n_state)
    
    # Now the analysis ensemble is:
    ens = ens_mean_new + a_a
    
    # Adaptive inflation based on spread matching
    ens_mean_after = np.mean(ens, axis=0)
    anomalies_after = ens - ens_mean_after
    cov_after = (anomalies_after.T @ anomalies_after) / (n_members - 1)
    
    # Expected innovation variance after update
    P_obs_obs_after = cov_after[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_after + R
    
    # Compute inflation factor using geometric mean of ratios
    ratios = np.ones(n_obs)
    for j in range(n_obs):
        exp_var = S_expected[j, j]
        if exp_var > 1e-10:
            ratios[j] = (innovation[j] ** 2 + observation_variance) / exp_var
    
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.95, 1.25))
    
    # Apply inflation around the new mean
    ens = ens_mean_after + inflation * (ens - ens_mean_after)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
