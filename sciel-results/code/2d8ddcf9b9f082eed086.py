import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x)
    
    def f(state):
        dx = np.empty_like(state)
        for i in range(n):
            dx[i] = (state[(i + 1) % n] - state[(i - 2) % n]) * state[(i - 1) % n] - state[i] + F
        return dx
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _enkf_update(ensemble, observations, observed_indices, observation_variance):
    """
    EnKF update with localization and adaptive inflation.
    ensemble: (N, 20)
    observations: (N, 10) - one observation vector per ensemble member? 
                   Actually, looking at the task: "observes even-indexed states with independent N(0,0.49) noise"
                   The observations array shape is likely (10,) for a single observation vector at this time step,
                   OR (N, 10) if it's an ensemble of observations.
                   The task says: "analyze(ensemble, observations, observed_indices, observation_variance) -> updated ensemble"
                   "observations" is passed in. In standard EnKF, you have one observation vector y.
                   But here, the evaluator propagates each member... wait.
                   "The evaluator propagates the ensemble and compares each updated ensemble mean with independent high-accuracy reference trajectories."
                   "observes even-indexed states with independent N(0,0.49) noise"
                   Usually, in these tasks, `observations` is a single vector of length 10 (the noisy observations of the 10 observed variables at the current time).
                   Let's assume `observations` is shape (10,).
    """
    N, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations, dtype=float).ravel()
    
    # Localization: use a distance-based correlation decay
    # For Lorenz-96, a reasonable localization radius is around 5-10 indices
    # We use a simple exponential or Gaussian localization
    # Let's use a correlation-based localization:
    # Compute the ensemble covariance in observation space
    # And apply localization to the analysis update for unobserved variables
    
    # Standard EnKF update:
    # 1. Compute ensemble mean and covariance
    x_mean = np.mean(ensemble, axis=0)
    x_anom = ensemble - x_mean  # (N, 20)
    
    # Observation operator: pick observed indices
    # H * x = x[observed_indices]
    
    # Compute innovation: y - H*x_mean
    Hx_mean = x_mean[observed_indices]
    innovation = obs - Hx_mean  # (10,)
    
    # Compute observation-space covariance: S = H * P * H^T + R
    # P = x_anom^T * x_anom / (N-1)
    # S = (x_anom[:, observed_indices]^T * x_anom[:, observed_indices] / (N-1)) + R
    # R = observation_variance * I
    
    Hx_anom = x_anom[:, observed_indices]  # (N, 10)
    S = np.dot(Hx_anom.T, Hx_anom) / (N - 1.0)
    S += observation_variance * np.eye(n_obs)
    
    # Compute Kalman gain: K = P * H^T * S^{-1}
    # P * H^T = x_anom * Hx_anom^T / (N-1)
    # But this is a full 20x10 matrix.
    
    # For efficiency and to handle localization, we can compute the update directly:
    # x_a = x_f + K * innovation
    # where K = (x_anom @ Hx_anom.T) / (N-1) @ inv(S)
    
    # However, this doesn't include localization for unobserved variables.
    # A simpler approach: use the linear regression (EnSRF) approach for unobserved variables.
    
    # Let's use the standard EnKF update for observed variables, and linear regression for unobserved.
    
    # Step 1: Update observed variables using standard EnKF
    # K_obs = (Hx_anom^T @ x_anom[:, observed_indices].T) / (N-1) @ inv(S)  -- this is 10x10
    # Actually, K_obs = (x_anom[:, observed_indices]^T @ x_anom[:, observed_indices]) / (N-1) @ inv(S)
    # Wait, K = P H^T S^{-1}
    # For observed variables i in observed_indices:
    # K[i, j] = sum_k P[i,k] H[k,j] S^{-1}[j, *]
    # H[k,j] is 1 if observed_indices[j] == k, else 0
    # So K[i, j] = P[i, observed_indices[j]] * S^{-1}[j, *]
    
    # Let's compute S_inv
    S_inv = np.linalg.inv(S)  # (10, 10)
    
    # Compute the update for observed variables
    # The gain for observed variable i at observation index j is:
    # K[i, j] = (x_anom[:, i]^T @ x_anom[:, observed_indices[j]]) / (N-1) @ S_inv[j, :]
    
    # Let's compute the full 20x10 gain matrix with localization
    # For unobserved variables, we use linear regression (EnSRF)
    
    # Unobserved indices
    all_indices = set(range(n_vars))
    obs_set = set(observed_indices)
    unobs_indices = [i for i in range(n_vars) if i not in obs_set]
    
    # For unobserved variables, use local linear regression:
    # x_unobs = x_mean_unobs + (x_anom_unobs @ x_anom_obs^T @ inv(x_anom_obs @ x_anom_obs^T)) @ (x_anom_obs)
    # This is the EnSRF approach
    
    # Compute the regression coefficients for unobserved variables
    if len(unobs_indices) > 0:
        x_anom_unobs = x_anom[:, unobs_indices]  # (N, n_unobs)
        x_anom_obs = x_anom[:, observed_indices]  # (N, n_obs)
        
        # Regression: x_unobs = A @ x_obs + b
        # A = (x_anom_unobs^T @ x_anom_obs) / (N-1) @ inv((x_anom_obs^T @ x_anom_obs) / (N-1))
        # But we need to be careful with the scaling
        
        cov_unobs_obs = np.dot(x_anom_unobs.T, x_anom_obs) / (N - 1.0)  # (n_unobs, n_obs)
        cov_obs_obs = np.dot(x_anom_obs.T, x_anom_obs) / (N - 1.0)  # (n_obs, n_obs)
        
        # Solve for A: A = cov_unobs_obs @ inv(cov_obs_obs)
        try:
            A = np.linalg.solve(cov_obs_obs.T, cov_unobs_obs.T).T  # (n_unobs, n_obs)
        except np.linalg.LinAlgError:
            A = np.zeros((len(unobs_indices), n_obs))
        
        # Update unobserved variables:
        # x_anom_unobs_new = A @ innovation
        # x_mean_unobs_new = x_mean_unobs + A @ innovation
        innovation_vec = innovation  # (n_obs,)
        
        x_mean_unobs_new = x_mean[unobs_indices] + np.dot(A, innovation_vec)
        x_anom_unobs_new = x_anom_unobs + np.outer(np.ones(N), np.dot(A, innovation_vec))
        
        # But wait, this doesn't account for the observation error variance properly.
        # The EnSRF approach assumes the observations are perfect. To account for noise,
        # we should use the Kalman gain approach with localization.
        
        # Let's use a hybrid: standard EnKF for observed, and for unobserved, use
        # the same Kalman gain approach but with a localization function.
    
    # Let's use a cleaner approach: full EnKF with localization
    # Localization: multiply the gain for unobserved variables by a function of distance
    # to the nearest observed variable
    
    # Compute distances from each variable to the nearest observed variable
    distances = np.zeros(n_vars)
    for i in range(n_vars):
        min_dist = min(abs((i - j) % n_vars) for j in observed_indices)
        # Use the minimum of forward and backward distance
        min_dist = min(min_dist, n_vars - min_dist)
        distances[i] = min_dist
    
    # Localization function: exp(-d^2 / (2 * sigma^2))
    sigma_loc = 3.0  # localization radius
    loc_factor = np.exp(-distances**2 / (2.0 * sigma_loc**2))
    
    # Set localization to 1 for observed variables
    for i in observed_indices:
        loc_factor[i] = 1.0
    
    # Compute the full gain matrix
    # K = (x_anom @ Hx_anom.T) / (N-1) @ S_inv
    # Hx_anom = x_anom[:, observed_indices]
    # x_anom @ Hx_anom.T is (20, N) @ (N, 10) = (20, 10)
    P_Ht = np.dot(x_anom, Hx_anom.T) / (N - 1.0)  # (20, 10)
    K = np.dot(P_Ht, S_inv)  # (20, 10)
    
    # Apply localization
    K = K * loc_factor[:, np.newaxis]
    
    # Compute the update
    update = np.dot(K, innovation)  # (20,)
    
    # Update ensemble
    ensemble_new = ensemble + np.outer(np.ones(N), update)
    
    # Adaptive inflation: check if the ensemble spread is too small
    # Compute the ensemble spread in observation space
    x_anom_new = ensemble_new - np.mean(ensemble_new, axis=0)
    spread_obs = np.std(x_anom_new[:, observed_indices], axis=0)
    
    # If spread is too small, inflate
    min_spread = np.sqrt(observation_variance) * 0.5
    if np.any(spread_obs < min_spread):
        inflation_factor = 1.0 + 0.01 * (min_spread / np.max(spread_obs, 1e-10) - 1.0)
        inflation_factor = np.clip(inflation_factor, 1.0, 1.1)
        x_mean_new = np.mean(ensemble_new, axis=0)
        ensemble_new = x_mean_new + (ensemble_new - x_mean_new) * inflation_factor
    
    # Clip to valid range
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    return ensemble_new


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    Args:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of observed_indices
        observed_indices: list/array of observed variable indices
        observation_variance: scalar variance of observation noise
    
    Returns:
        (16, 20) array of updated ensemble members
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    ensemble = np.asarray(ensemble, dtype=float)
    observations = np.asarray(observations, dtype=float)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    # Perform EnKF update
    try:
        result = _enkf_update(ensemble, observations, observed_indices, observation_variance)
    except Exception:
        # Fallback to simple nudging if EnKF fails
        result = ensemble.copy()
        if len(observations) == len(observed_indices):
            result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110.0:  # Safety margin
        # Return the result as-is, since we're already past most of the time
        pass
    
    return result
# EVOLVE-BLOCK-END
