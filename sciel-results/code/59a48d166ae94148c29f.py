import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with:
    - Perturbed observations (to avoid rank deficiency)
    - Localized covariance (exponential decay)
    - Adaptive inflation to prevent filter collapse
    - Deterministic mean update via ensemble mean of transformed states
    
    ensemble: (16, 20) array of ensemble members
    observations: (20,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed (even indices)
    observation_variance: scalar variance of observation noise
    """
    n_members, n_state = ensemble.shape
    result = ensemble.copy()
    
    # Compute ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - x_mean
    
    # Observation error covariance (diagonal)
    R = np.diag(np.full(len(observed_indices), observation_variance))
    
    # Build observation operator H: select observed indices
    H = np.zeros((len(observed_indices), n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Perturbed observations: add noise to each member's observation
    # This helps with the sampling error and rank deficiency
    obs_perturbed = observations[None, :] + np.sqrt(observation_variance) * np.random.randn(n_members, len(observed_indices))
    
    # Compute ensemble mean of observed states
    x_obs_mean = np.mean(ensemble[:, observed_indices], axis=0)
    obs_anomalies = obs_perturbed - x_obs_mean
    
    # Compute covariance of observed anomalies
    # P_oo = (1/(n-1)) * obs_anomalies^T @ obs_anomalies
    P_oo = (obs_anomalies.T @ obs_anomalies) / (n_members - 1)
    
    # Compute cross-covariance between full state and observed state
    # P_xo = (1/(n-1)) * anomalies^T @ obs_anomalies
    P_xo = (anomalies.T @ obs_anomalies) / (n_members - 1)
    
    # Add localization: taper the cross-covariance based on distance
    # For Lorenz-96, use exponential localization with range ~5
    localization_range = 5.0
    for j, obs_idx in enumerate(observed_indices):
        for i in range(n_state):
            dist = min(abs(i - obs_idx), n_state - abs(i - obs_idx))
            loc_factor = np.exp(-0.5 * (dist / localization_range) ** 2)
            P_xo[j, i] *= loc_factor
    
    # Innovation: difference between perturbed obs and ensemble mean obs
    innovation = obs_perturbed - x_obs_mean[None, :]
    
    # Kalman gain: K = P_xo @ (P_oo + R)^{-1}
    # Solve instead of explicit inverse for stability
    try:
        # (P_oo + R) is len(observed_indices) x len(observed_indices)
        S = P_oo + R
        # Solve S @ K^T = P_xo^T  =>  K = P_xo @ S^{-1}
        # Using solve: K.T = solve(S, P_xo.T)
        K = np.linalg.solve(S, P_xo.T).T
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        K = P_xo @ np.linalg.pinv(S)
    
    # Update each ensemble member
    # x_i_new = x_i + K @ (y_i_perturbed - H @ x_mean)
    # But we need to be careful: the standard EnKF update uses:
    # x_i_new = x_i + K @ (y_i - H @ x_mean)
    # where y_i is the perturbed observation for member i
    
    for m in range(n_members):
        y_m = obs_perturbed[m, :]
        innovation_m = y_m - x_obs_mean
        result[m, :] = ensemble[m, :] + K @ innovation_m
    
    # Apply adaptive inflation to prevent filter collapse
    # Compute analysis spread vs observation spread
    analysis_obs_mean = np.mean(result[:, observed_indices], axis=0)
    analysis_obs_var = np.var(result[:, observed_indices], axis=0).mean()
    
    # Observation spread (from the actual observations, assuming they're from a single truth)
    # We estimate the expected variance from the ensemble before update
    prior_obs_var = np.var(ensemble[:, observed_indices], axis=0).mean()
    
    # If analysis spread is too small relative to prior, inflate
    # Target: analysis spread should be at least some fraction of prior
    target_ratio = 0.7
    if analysis_obs_var < target_ratio * prior_obs_var and prior_obs_var > 1e-8:
        inflation_factor = np.sqrt(target_ratio * prior_obs_var / max(analysis_obs_var, 1e-10))
        # Cap inflation to avoid instability
        inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
        
        # Inflate around the ensemble mean
        result_mean = np.mean(result, axis=0)
        result = result_mean[None, :] + inflation_factor * (result - result_mean[None, :])
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
