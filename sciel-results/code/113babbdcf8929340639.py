"""Particle filter with kernel density for Lorenz-96 assimilation."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter update for Lorenz-96 ensemble assimilation.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations at even indices
        observed_indices: array of observed coordinate indices
        observation_variance: scalar variance of observation noise
    
    Returns:
        Updated (16, 20) ensemble
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n_members, n_dims = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    obs_var = float(observation_variance)
    
    if obs_var <= 0:
        obs_var = 0.49  # default
    
    # Step 1: Compute observation likelihoods for each particle
    # Use Gaussian kernel with adaptive bandwidth
    # Observation model: y_i ~ N(x_i, obs_var) for i in observed_indices
    
    # Extract observed components from each ensemble member
    # shape: (n_members, n_obs)
    obs_ensemble = ensemble[:, obs_idx]  # (16, 10)
    
    # Compute log-likelihood of observations given each particle
    # log L = -0.5 * sum_j [(y_j - x_j)^2 / obs_var + log(2*pi*obs_var)]
    diff = observations[None, :] - obs_ensemble  # (16, 10)
    log_lik = -0.5 * np.sum(diff ** 2 / obs_var, axis=1) - 0.5 * n_obs * np.log(2 * np.pi * obs_var)
    
    # Step 2: Compute importance weights
    # Use log-sum-exp for numerical stability
    max_log_lik = np.max(log_lik)
    weights = np.exp(log_lik - max_log_lik)
    weights /= np.sum(weights)
    
    # Step 3: Systematic resampling
    # Resample indices based on weights
    r = np.random.rand(n_members)
    positions = (r + np.arange(n_members)) / n_members
    cum_weights = np.cumsum(weights)
    
    resampled_indices = np.zeros(n_members, dtype=int)
    cum_idx = 0
    for i in range(n_members):
        while cum_idx < n_members - 1 and cum_weights[cum_idx] < positions[i]:
            cum_idx += 1
        resampled_indices[i] = cum_idx
    
    # Resampled ensemble
    resampled = ensemble[resampled_indices, :]
    
    # Step 4: Add small perturbation to maintain diversity (prevent collapse)
    # Use ensemble spread as scale
    mean_state = np.mean(resampled, axis=0)
    spread = np.std(resampled, axis=0)
    
    # Adaptive perturbation: small relative to local spread
    perturbation_scale = 0.05
    noise = perturbation_scale * spread * np.random.randn(n_members, n_dims)
    resampled = resampled + noise
    
    # Step 5: Ensemble inflation to prevent overconfidence
    # Compute current ensemble variance and inflate slightly
    current_var = np.var(resampled, axis=0)
    target_var = np.max(current_var, keepdims=True) * 0.1  # conservative target
    inflation_factor = 1.0 + 0.02  # small inflation
    
    # Apply inflation around the mean
    resampled = mean_state[None, :] + inflation_factor * (resampled - mean_state[None, :])
    
    # Step 6: Localized nudge toward observations for stability
    # This combines the particle filter result with a gentle nudge
    # to ensure we don't drift too far from observations
    nudge_strength = 0.15
    for i in range(n_members):
        for j, idx in enumerate(obs_idx):
            resampled[i, idx] += nudge_strength * (observations[j] - resampled[i, idx])
    
    # Step 7: Clip to valid range
    resampled = np.clip(resampled, -100.0, 100.0)
    
    # Ensure finite values
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return resampled
# EVOLVE-BLOCK-END
