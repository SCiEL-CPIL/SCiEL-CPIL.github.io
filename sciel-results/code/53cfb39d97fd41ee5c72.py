"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle Filter with:
    - Importance resampling (systematic)
    - Kernel density estimation for posterior modeling
    - Handles non-Gaussian tails of Lorenz-96 attractor
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = observation_variance
    obs_idx = np.asarray(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    # Copy ensemble
    result = np.array(ensemble, dtype=np.float64).copy()
    
    # --- Observation array ---
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 1:
        obs_arr = np.tile(obs_arr, (M, 1))
    
    # --- Compute likelihood weights for each particle ---
    # For each particle m and observation j, compute likelihood
    # p(y_j | x_m) = N(obs_j; x_m[j], obs_var)
    H_ens = result[:, obs_idx]  # (M, n_obs)
    
    # Compute log-likelihood for each particle
    # log p(y|x) = -0.5 * sum_j ( (y_j - x_j)^2 / obs_var + log(2*pi*obs_var) )
    innovation = obs_arr[0] - H_ens  # (M, n_obs) - use first observation (all same)
    log_weights = -0.5 * np.sum(innovation ** 2 / obs_var, axis=1) - 0.5 * n_obs * np.log(2 * np.pi * obs_var)
    
    # Normalize weights (softmax for numerical stability)
    log_weights_max = np.max(log_weights)
    log_weights -= log_weights_max
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # --- Systematic resampling ---
    # Determine resampling indices
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0  # ensure sum to 1
    
    # Systematic resampling: use uniform offset
    u0 = np.random.uniform(0, 1.0 / M)
    u = u0 + np.arange(M) / M
    u = np.minimum(u, 1.0)  # clip to [0, 1]
    
    # Find indices where u crosses cumulative weights
    resample_indices = np.searchsorted(cumulative_weights, u, side='right')
    resample_indices = np.clip(resample_indices, 0, M - 1)
    
    # Resampled ensemble
    resampled = result[resample_indices, :]  # (M, N)
    
    # --- Kernel Density Estimation (KDE) for posterior ---
    # For each dimension, use a Gaussian KDE with adaptive bandwidth
    # This helps capture non-Gaussian structure
    
    # Compute per-dimension standard deviation for bandwidth
    ens_std = np.std(resampled, axis=0, ddof=1)
    ens_std = np.maximum(ens_std, 1e-6)  # avoid zero
    
    # Silverman's rule of thumb for bandwidth
    bandwidths = 1.06 * ens_std * (M ** (-1.0 / 5.0))
    bandwidths = np.maximum(bandwidths, 0.1 * ens_std)
    
    # Compute KDE-based mean and covariance for posterior
    # For computational efficiency, we use a simplified approach:
    # The posterior mean is approximated by the weighted mean of resampled particles
    # with KDE smoothing
    
    # Compute posterior mean using KDE
    # For each dimension i, posterior_mean[i] = sum_m w_m * KDE_i(x_m)
    # But for efficiency, we use the resampled ensemble mean as the primary estimate
    # and apply KDE-based variance correction
    
    posterior_mean = np.mean(resampled, axis=0)
    
    # Compute KDE-based covariance estimate
    # For each pair (i, j), cov[i,j] = sum_m w_m * (x_m[i] - mean[i]) * (x_m[j] - mean[j])
    # with KDE smoothing
    
    # For efficiency, use the sample covariance with KDE bandwidth adjustment
    anomalies = resampled - posterior_mean  # (M, N)
    sample_cov = (anomalies.T @ anomalies) / (M - 1)  # (N, N)
    
    # Apply KDE-based variance inflation to account for non-Gaussian tails
    # The effective variance from KDE is larger than sample variance for heavy-tailed distributions
    kde_var_factor = 1.0 + 0.2 * (N / M)  # mild inflation based on dimensionality
    posterior_cov = sample_cov * kde_var_factor
    
    # --- Generate new ensemble from KDE posterior ---
    # Use the posterior mean and covariance to generate new particles
    # This combines the resampled particles with KDE-based reconstruction
    
    # Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(posterior_cov + 1e-8 * np.eye(N))
    except np.linalg.LinAlgError:
        L = np.eye(N) * np.sqrt(np.mean(np.diag(posterior_cov)))
    
    # Generate new ensemble: blend resampled particles with KDE samples
    # This maintains ensemble diversity while incorporating observation information
    
    # Mix: 70% resampled + 30% KDE samples
    mix_ratio = 0.7
    
    # KDE samples
    std_normal = np.random.randn(M, N)
    kde_samples = posterior_mean[None, :] + std_normal @ L.T  # (M, N)
    
    # Blend
    new_ensemble = mix_ratio * resampled + (1 - mix_ratio) * kde_samples
    
    # --- Adaptive spread correction ---
    # Ensure ensemble spread is not too small
    new_ens_std = np.std(new_ensemble, axis=0, ddof=1)
    min_std = 0.5 * np.sqrt(obs_var)
    
    inflation = np.ones(N)
    for i in range(N):
        if new_ens_std[i] < min_std:
            inflation[i] = min_std / max(new_ens_std[i], 1e-10)
            inflation[i] = min(inflation[i], 2.0)
    
    # Apply inflation
    new_ens_mean = np.mean(new_ensemble, axis=0)
    new_ensemble = new_ens_mean[None, :] + inflation[None, :] * (new_ensemble - new_ens_mean[None, :])
    
    # Safety: clip to valid range
    np.clip(new_ensemble, -100.0, 100.0, out=new_ensemble)
    
    # Ensure finite
    if not np.all(np.isfinite(new_ensemble)):
        new_ensemble = np.array(ensemble, dtype=np.float64).copy()
        new_ensemble[:, obs_idx] += 0.5 * (obs_arr[0] - new_ensemble[:, obs_idx])
        np.clip(new_ensemble, -100.0, 100.0, out=new_ensemble)
    
    return new_ensemble

# EVOLVE-BLOCK-END
