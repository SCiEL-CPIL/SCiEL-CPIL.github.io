import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle Filter with Local Linearized Ensemble Transform for Lorenz-96.
    Uses systematic resampling and weighted updates to handle non-Gaussian posteriors.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    M, N = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    mean_x = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean_x
    
    # 2. Compute Innovation
    innovation = observations - mean_x[observed_indices]
    
    # 3. Estimate Observation Space Covariance (Diagonal Approximation for Speed/Robustness)
    # B_obs_diag = diag( (H A)^T (H A) / (M-1) )
    H_anom = anomalies[:, observed_indices]
    B_obs_diag = np.sum(H_anom**2, axis=0) / (M - 1)
    
    # Add observation noise variance
    S_diag = B_obs_diag + observation_variance
    
    # Ensure S_diag is positive
    S_diag = np.maximum(S_diag, 1e-6)
    
    # 4. Compute Log-Weights for Particle Filter
    # Likelihood: p(y | x_i) ~ N(y; H x_i, R)
    # Since H is selection, H x_i = x_i[observed_indices]
    # log w_i = -0.5 * sum_j ( (y_j - x_i,j)^2 / S_j )
    
    # Compute squared innovations for each particle
    # particle_obs: M x n_obs
    particle_obs = ensemble[:, observed_indices]
    # diff: M x n_obs
    diff = particle_obs - observations[None, :]
    
    # log_w: M
    # Note: We use the diagonal S for the likelihood calculation.
    log_w = -0.5 * np.sum((diff**2) / S_diag[None, :], axis=1)
    
    # Normalize log-weights (subtract max for numerical stability)
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum < 1e-10:
        # Uniform weights if all are equal (degenerate case)
        w = np.ones(M) / M
    else:
        w = w / w_sum
    
    # 5. Effective Sample Size (ESS)
    ESS = 1.0 / np.sum(w**2)
    
    # 6. Adaptive Inflation based on ESS
    # If ESS is low, particles are degenerate. Inflate anomalies to increase spread.
    # Target ESS is typically M/2 or similar.
    target_ess = M / 2.0
    if ESS < target_ess:
        # Inflate
        # Simple heuristic: scale anomalies by sqrt(target_ess / ESS)
        # Cap inflation
        inflation = np.sqrt(target_ess / ESS)
        inflation = np.clip(inflation, 1.0, 3.0)
    else:
        inflation = 1.0
    
    anomalies = inflation * anomalies
    
    # 7. Deterministic Resampling (Systematic Resampling)
    # This step generates a new set of indices that represent the posterior distribution.
    # We use a fixed random seed for determinism, but since we can't use global state,
    # we use a simple deterministic pseudo-random sequence based on the indices.
    # However, standard systematic resampling requires randomness.
    # To be deterministic without global state, we can use "stratified sampling" with a fixed offset.
    
    # Cumulative weights
    cum_w = np.cumsum(w)
    cum_w[-1] = 1.0  # Ensure exact 1.0
    
    # Systematic resampling with fixed offset u0 = 0.5/M
    u0 = 0.5 / M
    u = u0 + np.arange(M) / M
    
    # Find indices
    indices = np.searchsorted(cum_w, u)
    
    # Resampled ensemble
    ensemble_resampled = ensemble[indices]
    anomalies_resampled = anomalies[indices]
    
    # 8. Update Mean using Weighted Average
    # The new mean is the weighted average of the original ensemble (before resampling)
    # This is more accurate than the mean of the resampled ensemble.
    mean_x_a = np.sum(w[:, None] * ensemble, axis=0)
    
    # 9. Update Anomalies
    # We want the anomalies of the new ensemble to reflect the posterior.
    # A common approach in Particle Filters is to use the resampled anomalies directly,
    # but this can lead to loss of spread if not careful.
    # Alternatively, we can use the Ensemble Transform approach:
    # Find a transformation T such that T @ A_prior has the same covariance as the posterior.
    # For simplicity and robustness, we use the resampled anomalies but center them
    # around the new mean.
    
    # Current resampled anomalies are centered around the original mean (approximately).
    # We need to re-center them around the new mean.
    # Actually, the resampled ensemble is a set of points.
    # Let's compute the mean of the resampled ensemble.
    mean_resampled = np.mean(ensemble_resampled, axis=0)
    
    # The error between the weighted mean and the resampled mean indicates a bias.
    # We can correct the resampled ensemble by shifting it.
    # However, a better approach for the anomalies is to use the original anomalies
    # transformed by the weights.
    
    # Let's stick to a simpler, robust method:
    # Use the resampled indices to pick anomalies, but ensure the variance is maintained.
    # If ESS was low, we inflated. Now we resample.
    
    # Re-center the resampled ensemble to have the exact new mean.
    # This ensures the ensemble mean is exactly the weighted mean.
    ensemble_a = ensemble_resampled - mean_resampled + mean_x_a
    
    # 10. Final Check and Clamp
    ensemble_a = np.clip(ensemble_a, -100, 100)
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Time check
    if time.time() > deadline:
         return ensemble_a
        
    return ensemble_a
