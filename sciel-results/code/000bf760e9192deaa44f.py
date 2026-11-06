"""Nonlinear particle filter with resampling and kernel density estimation for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear particle filter update with importance resampling and KDE.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations at even indices
    - observed_indices: (10,) array of observed variable indices
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Ensure inputs are float arrays
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    
    # Compute ensemble statistics
    ens_mean = np.mean(ensemble, axis=0)
    ens_anom = ensemble - ens_mean[None, :]
    
    # Compute ensemble covariance for bandwidth estimation
    ens_cov = (ens_anom.T @ ens_anom) / (n_members - 1)
    
    # Step 1: Compute importance weights based on observation likelihood
    # Gaussian likelihood: p(y|x) ~ N(Hx, R)
    # log_weight = -0.5 * ||y - Hx||^2 / R
    obs_var = observation_variance
    weights = np.zeros(n_members)
    
    for i in range(n_members):
        # Innovation for member i
        innov = observations - ensemble[i, obs_idx]
        # Log-likelihood (up to additive constant)
        weights[i] = -0.5 * np.sum(innov**2) / obs_var
    
    # Normalize weights (with numerical stability)
    weights -= np.max(weights)
    weights = np.exp(weights)
    weights /= np.sum(weights)
    
    # Step 2: Systematic resampling
    # Resample n_members particles from the weighted distribution
    u0 = np.random.uniform(0, 1.0 / n_members)
    u = u0 + np.arange(n_members) / n_members
    
    # Cumulative weights
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0  # Ensure last value is exactly 1
    
    resample_indices = np.searchsorted(cum_weights, u)
    resample_indices = np.clip(resample_indices, 0, n_members - 1)
    
    # Resampled ensemble
    resampled_ensemble = ensemble[resample_indices, :]
    
    # Step 3: Kernel density estimation to create posterior ensemble
    # Use a Gaussian KDE with bandwidth based on ensemble spread
    # For each dimension, estimate the posterior as a mixture of Gaussians
    # centered at resampled particles
    
    # Estimate bandwidth per dimension using Silverman's rule
    # h_j = (4n+2)^(-1/6) * std_j for Gaussian KDE
    ens_std = np.std(resampled_ensemble, axis=0)
    bandwidth = 0.9 * np.minimum(ens_std, np.sqrt(ens_cov.diagonal())) * n_members ** (-1.0 / (n_vars + 4))
    bandwidth = np.maximum(bandwidth, 0.01)  # Minimum bandwidth
    
    # Generate new ensemble by sampling from the KDE
    # For efficiency, we'll use a deterministic approach:
    # Create the new ensemble by blending resampled particles with perturbations
    # based on the KDE structure
    
    # Compute the posterior mean and covariance from resampled ensemble
    post_mean = np.mean(resampled_ensemble, axis=0)
    post_anom = resampled_ensemble - post_mean[None, :]
    post_cov = (post_anom.T @ post_anom) / (n_members - 1)
    
    # Generate new ensemble using Cholesky decomposition of posterior covariance
    # This gives us a Gaussian approximation of the KDE
    try:
        L = np.linalg.cholesky(post_cov)
    except np.linalg.LinAlgError:
        # Fallback: add small regularization
        post_cov_reg = post_cov + 1e-6 * np.eye(n_vars)
        try:
            L = np.linalg.cholesky(post_cov_reg)
        except np.linalg.LinAlgError:
            # Last resort: use diagonal
            L = np.diag(np.sqrt(np.maximum(np.diag(post_cov), 1e-6)))
    
    # Generate standard normal random numbers
    z = np.random.randn(n_members, n_vars)
    
    # Create new ensemble: post_mean + L @ z^T
    # But we need to ensure this is consistent with the observation constraints
    # Blend the Gaussian KDE sample with the resampled particles
    
    kde_ensemble = post_mean[None, :] + z @ L.T
    
    # Blend with resampled particles to maintain non-Gaussian features
    # Use a mixing coefficient based on how well the Gaussian fits
    # Compute the Mahalanobis distance of observations
    H = np.zeros((n_obs, n_vars))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    # Compute how well the Gaussian posterior explains the observations
    obs_mean = H @ post_mean
    obs_cov = H @ post_cov @ H.T + np.eye(n_obs) * obs_var
    
    try:
        obs_inv = np.linalg.inv(obs_cov)
    except np.linalg.LinAlgError:
        obs_inv = np.linalg.pinv(obs_cov)
    
    innov = observations - obs_mean
    mahalanobis_sq = innov @ obs_inv @ innov
    
    # If Mahalanobis distance is large, trust the resampled particles more
    # If small, trust the Gaussian KDE more
    # Blend coefficient: alpha = 1 / (1 + mahalanobis_sq / n_obs)
    alpha = 1.0 / (1.0 + mahalanobis_sq / max(n_obs, 1))
    alpha = np.clip(alpha, 0.3, 0.7)  # Keep blend in reasonable range
    
    # Final ensemble: blend of KDE and resampled
    updated_ensemble = alpha * kde_ensemble + (1 - alpha) * resampled_ensemble
    
    # Ensure the updated ensemble is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Replace any non-finite values with the posterior mean
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, post_mean)
    
    return updated_ensemble
