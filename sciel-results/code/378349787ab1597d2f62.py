import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Relaxed Ensemble Smoother with Localization for Lorenz-96.
    This approach uses perturbed observations and a relaxation parameter
    to update the ensemble, distinct from the standard EnKF mean-update structure.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute Forecast Statistics
    x_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - x_mean
    # Sample covariance
    S = (anomalies.T @ anomalies) / (n_members - 1)
    
    # 2. Observation Operator Terms
    S_HH = S[np.ix_(observed_indices, observed_indices)]
    S_xH = S[:, observed_indices]
    
    # Add observation noise
    S_HH = S_HH + observation_variance * np.eye(n_obs)
    
    # 3. Compute Gain
    try:
        S_HH_inv = np.linalg.inv(S_HH)
        K = S_xH @ S_HH_inv
    except np.linalg.LinAlgError:
        S_HH_inv = np.linalg.pinv(S_HH)
        K = S_xH @ S_HH_inv
        
    # 4. Localize the Gain
    L_loc = 3.0 # Localization length scale
    
    def ring_distance(i, j, n):
        d = abs(i - j)
        return min(d, n - d)
        
    K_loc = np.zeros_like(K)
    for i in range(n_vars):
        for j in range(n_obs):
            d = ring_distance(i, observed_indices[j], n_vars)
            w = np.exp(-0.5 * (d / L_loc) ** 2)
            K_loc[i, j] = K[i, j] * w
            
    # 5. Relaxed Ensemble Smoother Update
    alpha = 0.8 # Relaxation parameter
    
    # Perturb observations with a fixed seed for determinism
    rng = np.random.RandomState(12345)
    perturbations = rng.normal(0, np.sqrt(observation_variance), size=(n_members, n_obs))
    y_pert = observations[np.newaxis, :] + perturbations
    
    # Update each member
    x_new = np.zeros_like(ensemble)
    for k in range(n_members):
        innovation = y_pert[k] - x_mean[observed_indices]
        correction = K_loc @ innovation
        x_new[k] = ensemble[k] + alpha * correction
        
    # 6. Post-update inflation to maintain spread
    x_new_mean = np.mean(x_new, axis=0)
    x_new = x_new_mean + (x_new - x_new_mean) * 1.02
    
    # 7. Clip and check finiteness
    x_new = np.clip(x_new, -100.0, 100.0)
    x_new = np.where(np.isfinite(x_new), x_new, 0.0)
    
    return x_new
