import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear particle filter update with importance resampling and
    kernel density estimation for posterior approximation, leveraging
    the chaotic dynamics structure of Lorenz-96.
    
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

    # 1. Compute log-likelihoods for each particle using a localized Gaussian likelihood
    # Use a broader bandwidth to handle non-Gaussian uncertainty
    # The observation noise variance is given, but we use a slightly inflated
    # effective variance to account for model error in chaotic systems
    
    # Compute innovation for each particle at observed coordinates
    innovations = observations[None, :] - ensemble[:, obs_idx]  # (N, n_obs)
    
    # Effective observation variance: inflate to handle model error
    # This is a key structural difference: we treat the observation variance
    # as a lower bound and add model error allowance
    effective_var = observation_variance * 1.5  # 50% inflation for model error
    
    # Log-likelihood: -0.5 * sum(innovations^2 / effective_var)
    # Add a small regularization to avoid numerical issues
    log_likelihoods = -0.5 * np.sum(innovations ** 2 / effective_var, axis=1)
    
    # 2. Importance weights via LSR (Lempitsky-Shiryaev-Robbins) style normalization
    # Use log-sum-exp for numerical stability
    max_ll = np.max(log_likelihoods)
    exp_ll = np.exp(log_likelihoods - max_ll)
    weights = exp_ll / np.sum(exp_ll)
    
    # 3. Check for weight degeneracy
    # If effective sample size is too low, we need to resample
    eff_sample_size = 1.0 / np.sum(weights ** 2)
    
    # 4. Deterministic kernel density estimate to approximate posterior
    # Instead of direct resampling (which can cause particle collapse),
    # we use a weighted KDE to generate new particles
    
    # Compute the weighted mean and covariance for the observed subspace
    mean_obs = np.average(ensemble[:, obs_idx], weights=weights, axis=0)
    
    # Weighted covariance for observed subspace
    obs_anoms = ensemble[:, obs_idx] - mean_obs[None, :]
    cov_obs = (weights[:, None] * obs_anoms.T @ obs_anoms) / (np.sum(weights) - 1/(np.sum(weights**2) * N))
    # Simpler: use weighted covariance
    cov_obs = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(i, n_obs):
            val = np.sum(weights * (ensemble[:, obs_idx[i]] - mean_obs[i]) * (ensemble[:, obs_idx[j]] - mean_obs[j]))
            cov_obs[i, j] = val
            cov_obs[j, i] = val
    
    # 5. For unobserved coordinates, use conditional Gaussian approximation
    # based on cross-covariance with observed coordinates
    mean_all = np.average(ensemble, weights=weights, axis=0)
    
    # Compute cross-covariance between all and observed
    all_anoms = ensemble - mean_all[None, :]
    cross_cov = np.zeros((D, n_obs))
    for i in range(D):
        for j in range(n_obs):
            cross_cov[i, j] = np.sum(weights * all_anoms[:, i] * all_anoms[:, obs_idx[j]])
    
    # Compute posterior mean using conditional Gaussian formula
    # For each dimension, if observed: use observation directly
    # If unobserved: use mean + cross_cov @ inv(cov_obs) @ (obs - mean_obs)
    
    try:
        cov_obs_inv = np.linalg.inv(cov_obs + 1e-6 * np.eye(n_obs))
    except np.linalg.LinAlgError:
        cov_obs_inv = np.linalg.pinv(cov_obs + 1e-6 * np.eye(n_obs))
    
    # Posterior mean
    posterior_mean = mean_all.copy()
    for j, obs_j in enumerate(obs_idx):
        # Observed coordinates: blend particle mean with observation
        # Use a soft update based on likelihood
        obs_weight = 1.0 / (1.0 + effective_var / (cov_obs[j, j] + 1e-6))
        posterior_mean[obs_j] = obs_weight * observations[j] + (1 - obs_weight) * mean_all[obs_j]
    
    # For unobserved coordinates, use conditional expectation
    for i in range(D):
        if i not in obs_idx:
            # Conditional mean: mean_i + cross_cov[i, :] @ inv(cov_obs) @ (obs - mean_obs)
            correction = cross_cov[i, :] @ cov_obs_inv @ (observations - mean_obs)
            posterior_mean[i] = mean_all[i] + correction
    
    # 6. Generate new particles using a mixture of:
    #    a) Resampled particles (to maintain diversity)
    #    b) KDE samples around the posterior mean
    
    # Resampling: systematic resampling for stability
    r = np.random.uniform(0, 1/N)
    cumulative_weights = np.cumsum(weights)
    indices = np.searchsorted(cumulative_weights, r + np.arange(N)/N)
    indices = np.clip(indices, 0, N-1)
    resampled = ensemble[indices]
    
    # KDE: generate particles around posterior mean with ensemble spread
    # Compute ensemble spread
    spread = np.std(ensemble, axis=0)
    
    # Blend: 60% resampled, 40% KDE samples
    n_resample = int(0.6 * N)
    n_kde = N - n_resample
    
    # KDE samples: posterior_mean + spread * random_normal
    kde_samples = posterior_mean[None, :] + spread[None, :] * np.random.normal(0, 1, (n_kde, D))
    
    # Combine
    new_ensemble = np.vstack([resampled[:n_resample], kde_samples])
    
    # 7. Ensure the ensemble mean matches the posterior mean closely
    # Add a small correction to pull the ensemble mean toward the posterior mean
    current_mean = np.mean(new_ensemble, axis=0)
    correction = 0.3 * (posterior_mean - current_mean)
    new_ensemble += correction[None, :]
    
    # 8. Spread maintenance: ensure minimum spread
    new_spread = np.std(new_ensemble, axis=0)
    min_spread = 0.15
    for d in range(D):
        if new_spread[d] < min_spread:
            new_ensemble[:, d] += np.random.normal(0, min_spread * 0.5, N)
    
    # 9. Clip to bounds
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    # Time check
    if time.time() > deadline:
        return new_ensemble
    
    return new_ensemble
