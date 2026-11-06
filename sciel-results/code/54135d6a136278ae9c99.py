import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Two-Pass Localized EnKF update with adaptive inflation
    and spread-preserving mechanisms for Lorenz-96 assimilation.
    
    Strategy:
    1. First Pass: Localized EnKF with narrow length scale (L=2.0) for tight local corrections
    2. Second Pass: Localized EnKF with wider length scale (L=4.0) for global consistency
    3. Adaptive inflation based on spread preservation in observed and unobserved components
    4. Clip to ensure finite, bounded output
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    def _localized_enkf_pass(X, y, obs_indices, obs_var, L):
        """Single localized EnKF pass with exponential localization."""
        N_m, D_m = X.shape
        n_o = len(obs_indices)
        
        mean = np.mean(X, axis=0)
        anomalies = X - mean
        
        # Exponential localization weights
        loc_weights = np.zeros((D_m, n_o))
        for i in range(D_m):
            for j, oj in enumerate(obs_indices):
                dist = abs(i - oj)
                dist = min(dist, D_m - dist)
                loc_weights[i, j] = np.exp(-dist / L)
        
        # Observation anomalies
        anomalies_obs = anomalies[:, obs_indices]
        
        # Observation covariance
        C_oo = (anomalies_obs.T @ anomalies_obs) / (N_m - 1)
        R = obs_var * np.eye(n_o)
        S = C_oo + R
        
        # State-observation covariance with localization
        C_xo = (anomalies.T @ anomalies_obs) / (N_m - 1)
        C_xo_loc = C_xo * loc_weights
        
        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = C_xo_loc @ S_inv
        
        # Update
        innovation = y - mean[obs_indices]
        mean_new = mean + K @ innovation
        
        x_obs = X[:, obs_indices]
        innovations_ens = y[None, :] - x_obs
        X_new = X + (K @ innovations_ens.T).T
        
        return X_new
    
    # Pass 1: Narrow localization
    X1 = _localized_enkf_pass(ensemble, observations, obs_idx, observation_variance, L=2.0)
    
    # Check time
    if time.time() > deadline:
        return np.clip(X1, -100.0, 100.0)
    
    # Pass 2: Wider localization on the result of pass 1
    X2 = _localized_enkf_pass(X1, observations, obs_idx, observation_variance, L=4.0)
    
    # Check time
    if time.time() > deadline:
        return np.clip(X2, -100.0, 100.0)
    
    # Adaptive inflation: preserve pre-update spread
    spread_pre = np.std(ensemble, axis=0)
    mean_post = np.mean(X2, axis=0)
    spread_post = np.std(X2, axis=0)
    
    inflation_factor = np.ones(D)
    for d in range(D):
        if spread_post[d] > 1e-6 and spread_pre[d] > 1e-6:
            ratio = spread_pre[d] / spread_post[d]
            if ratio > 1.0:
                inflation_factor[d] = 1.0 + 0.6 * (ratio - 1.0)
    
    X2 = X2 - mean_post
    X2 *= inflation_factor
    X2 += mean_post
    
    return np.clip(X2, -100.0, 100.0)
