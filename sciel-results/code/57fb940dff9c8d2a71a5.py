"""Ensemble Square Root Filter with regression-based update for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter (EnSRF) analysis update.
    
    1. Observed coordinates are updated using the standard Kalman gain formula,
       adjusting the mean and scaling the anomalies.
    2. Unobserved coordinates are updated by regressing their forecast anomalies
       onto the forecast anomalies of the observed coordinates. The change in
       the observed anomalies (due to the update) is projected onto the
       unobserved coordinates to update their anomalies. The means are updated
       similarly.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed coordinates (even-indexed).
    observed_indices : array-like
        Indices of the observed coordinates.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    X = ensemble.copy().astype(np.float64)
    n_ens, n_state = X.shape
    
    obs_idx_arr = np.asarray(observed_indices, dtype=int)
    obs = np.asarray(observations, dtype=np.float64)
    R = float(observation_variance)
    
    # Create a set for fast lookup of observed indices
    obs_idx_set = set(obs_idx_arr.tolist())
    
    # Store forecast means and anomalies for all variables
    X_forecast = X.copy()
    means_f = np.mean(X_forecast, axis=0)
    anoms_f = X_forecast - means_f[np.newaxis, :]
    
    # Initialize analysis state
    X_analysis = X_forecast.copy()
    means_a = means_f.copy()
    anoms_a = anoms_f.copy()
    
    # 1. Update Observed Coordinates
    for k, idx in enumerate(obs_idx_arr):
        idx = int(idx)
        var_j = np.var(anoms_f[:, idx])
        
        if var_j < 1e-12:
            # Degenerate case: set to observation
            means_a[idx] = obs[k]
            anoms_a[:, idx] = 0.0
            continue
            
        K_j = var_j / (var_j + R)
        resid = obs[k] - means_f[idx]
        
        means_a[idx] = means_f[idx] + K_j * resid
        anoms_a[:, idx] = anoms_f[:, idx] * np.sqrt(1.0 - K_j)
        
        # Reconstruct X_analysis for this column
        X_analysis[:, idx] = means_a[idx] + anoms_a[:, idx]
        
    # 2. Update Unobserved Coordinates via Regression
    # We regress the forecast anomalies of unobserved variables onto the
    # forecast anomalies of observed variables.
    
    # Extract forecast anomalies for observed variables
    A_obs_f = anoms_f[:, obs_idx_arr] # shape (n_ens, n_obs)
    
    # Calculate the change in anomalies for observed variables
    delta_anoms_obs = anoms_a[:, obs_idx_arr] - A_obs_f # shape (n_ens, n_obs)
    
    # Calculate the change in means for observed variables
    delta_means_obs = means_a[obs_idx_arr] - means_f[obs_idx_arr] # shape (n_obs,)
    
    # For each unobserved coordinate, compute regression coefficients
    # and apply the update
    for j in range(n_state):
        if j in obs_idx_set:
            continue
            
        # Get forecast anomalies for unobserved variable j
        a_j_f = anoms_f[:, j]
        
        # Solve for regression coefficients beta such that:
        # a_j_f ≈ A_obs_f @ beta
        # Using least squares
        try:
            beta, _, _, _ = np.linalg.lstsq(A_obs_f, a_j_f, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(A_obs_f.shape[1])
            
        # Calculate the change in anomalies for j
        # delta_a_j = delta_anoms_obs @ beta
        delta_a_j = delta_anoms_obs @ beta
        
        # Calculate the change in mean for j
        # delta_mu_j = delta_means_obs @ beta
        delta_mu_j = delta_means_obs @ beta
        
        # Update the analysis state for j
        # x_j_a = x_j_f + delta_mu_j + delta_a_j
        # Note: x_j_f = means_f[j] + a_j_f
        # So x_j_a = means_f[j] + delta_mu_j + a_j_f + delta_a_j
        #          = (means_f[j] + delta_mu_j) + (a_j_f + delta_a_j)
        #          = means_a[j] + anoms_a[:, j]
        
        means_a[j] = means_f[j] + delta_mu_j
        anoms_a[:, j] = a_j_f + delta_a_j
        X_analysis[:, j] = means_a[j] + anoms_a[:, j]
        
    # Clip to valid range and ensure finiteness
    X_analysis = np.clip(X_analysis, -100.0, 100.0)
    X_analysis = np.where(np.isfinite(X_analysis), X_analysis, 0.0)
    
    return X_analysis
# EVOLVE-BLOCK-END
