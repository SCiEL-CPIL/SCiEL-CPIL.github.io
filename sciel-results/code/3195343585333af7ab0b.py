import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Particle Filter update with LSR weights and resampling.
    This approach treats the ensemble as particles and uses likelihood-based
    weighting to handle non-Gaussian uncertainties inherent in chaotic systems
    like Lorenz-96, which standard linear EKF updates may misestimate.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array with shape (16, 20).
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

    # 1. Compute Log-Likelihoods for each particle
    # LSR (Lempitsky-Shiryaev-Robust) weights are often more stable than 
    # standard Gaussian likelihoods in high-dimensional or chaotic systems 
    # because they are less sensitive to outliers and heavy tails.
    # However, standard Gaussian log-likelihood is often sufficient if the 
    # ensemble spread is well-calibrated. Let's use a robust variant:
    # w_i = exp(-0.5 * sum((y - Hx_i)^2 / R))
    # To avoid underflow, we subtract the max log-likelihood.
    
    x_obs = ensemble[:, obs_idx] # (N, n_obs)
    innov = observations[None, :] - x_obs # (N, n_obs)
    
    # Gaussian log-likelihood
    log_likelihoods = -0.5 * np.sum((innov ** 2) / observation_variance, axis=1)
    
    # Normalize weights
    max_log_l = np.max(log_likelihoods)
    weights = np.exp(log_likelihoods - max_log_l)
    weights /= np.sum(weights)
    
    # 2. Check for weight degeneracy
    # If effective sample size is too low, we need to resample.
    # ESS = 1 / sum(w^2)
    ess = 1.0 / np.sum(weights ** 2)
    
    # 3. Resampling and Update
    # We use systematic resampling to select new particles.
    # This effectively "moves" the ensemble towards the more likely regions.
    
    # To maintain the ensemble spread and avoid collapse, we can add a small 
    # perturbation or use a "perturbed" resampling scheme. 
    # However, simple systematic resampling is standard for PFs.
    
    # Systematic resampling
    r = np.random.uniform(0, 1.0, N)
    u = (r + np.arange(N)) / N
    c = np.cumsum(weights)
    indices = np.zeros(N, dtype=int)
    i = 0
    for j in range(N):
        while u[j] > c[i]:
            i += 1
        indices[j] = i
    
    # Resampled ensemble
    new_ensemble = ensemble[indices, :]
    
    # 4. Spread Maintenance / Inflation
    # Resampling reduces spread. We need to reinflate to avoid filter collapse.
    # A common method is to add Gaussian noise scaled by the estimated spread.
    # We estimate the spread from the pre-resampling ensemble or a prior.
    
    # Calculate pre-resampling spread (standard deviation of anomalies)
    mean_pre = np.mean(ensemble, axis=0)
    spread_pre = np.std(ensemble, axis=0)
    
    # Calculate post-resampling spread
    mean_post = np.mean(new_ensemble, axis=0)
    spread_post = np.std(new_ensemble, axis=0)
    
    # Inflation factor: scale anomalies to match pre-resampling spread
    # This is a simple heuristic to maintain diversity.
    # We want spread_post to be roughly equal to spread_pre.
    # new_anomalies = (new_ensemble - mean_post) * (spread_pre / spread_post)
    # But we must be careful with zero spread.
    
    inflation_factor = np.ones(D)
    for d in range(D):
        if spread_post[d] > 1e-6:
            inflation_factor[d] = spread_pre[d] / spread_post[d]
        else:
            # If spread collapsed, add random noise
            inflation_factor[d] = 1.0
            
    # Apply inflation
    new_ensemble = new_ensemble - mean_post
    new_ensemble *= inflation_factor
    new_ensemble += mean_post
    
    # 5. Optional: Localized Nudging
    # Pure particle filters can struggle with high-dimensional localization.
    # We can add a small localized EKF correction to the resampled particles
    # to improve accuracy in observed regions.
    
    # Compute localized EKF gain
    A_obs = ensemble[:, obs_idx]
    C_oo = (A_obs.T @ A_obs) / (N - 1)
    R = observation_variance * np.eye(n_obs)
    S = C_oo + R
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    A_all = ensemble - np.mean(ensemble, axis=0)
    C_xo = (A_all.T @ A_obs) / (N - 1)
    
    # Localization weights
    loc_weights = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)
            loc_weights[i, j] = np.exp(-dist / 3.5)
            
    C_xo_loc = C_xo * loc_weights
    K = C_xo_loc @ S_inv
    
    # Apply a small fraction of the EKF update to the particles
    # This combines the robustness of PF with the accuracy of EKF.
    # alpha = 0.5 means we take half the EKF correction.
    alpha = 0.5
    for i in range(N):
        x_i_obs = new_ensemble[i, obs_idx]
        corr = K @ (observations - x_i_obs)
        new_ensemble[i] += alpha * corr
        
    # 6. Clip to bounds
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    # Time check
    if time.time() > deadline:
        return new_ensemble
    
    return new_ensemble
