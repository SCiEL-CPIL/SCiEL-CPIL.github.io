import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Hybrid Ensemble Particle Filter (EnPF) update.
    
    Strategy:
    1. Compute Gaussian log-likelihoods for each ensemble member (particle).
    2. Normalize weights using the Log-Sum-Exp trick for numerical stability.
    3. Apply systematic resampling to generate a new ensemble of equal weights.
    4. Apply deterministic spread restoration (inflation) to maintain ensemble variance
       relative to the observation error, preventing collapse in chaotic systems.
    5. Clip to bounds.
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

    # 1. Compute Log-Likelihoods
    # We assume independent Gaussian observation errors.
    # log p(y | x_i) = -0.5 * sum_j [ (y_j - H x_i)_j^2 / R_jj ] + const
    # Since R is scalar variance, we can omit the constant term for weighting.
    
    # Extract observed values from each member
    # shape: (N, n_obs)
    H_ensemble = ensemble[:, obs_idx]
    
    # Residuals: (N, n_obs)
    residuals = observations[None, :] - H_ensemble
    
    # Squared residuals: (N, n_obs)
    sq_res = residuals ** 2
    
    # Log-likelihoods: (N,)
    # Assuming uniform observation variance
    log_likelihoods = -0.5 * np.sum(sq_res, axis=1) / observation_variance

    # 2. Normalize Weights (LSR)
    max_log_l = np.max(log_likelihoods)
    log_weights = log_likelihoods - max_log_l
    weights = np.exp(log_weights)
    
    # Handle case where all weights are zero (underflow) or NaN
    if np.sum(weights) == 0 or not np.all(np.isfinite(weights)):
        # Fallback to uniform weights
        weights = np.ones(N) / N
    else:
        weights /= np.sum(weights)

    # 3. Systematic Resampling
    # Generate N random numbers with spacing 1/N, offset by a single uniform random number
    r = np.random.uniform(0, 1.0 / N)
    u = r + np.arange(N) / N
    
    # Cumulative sum of weights
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # Ensure it ends exactly at 1.0
    
    # Find indices
    # np.searchsorted returns the index where u[i] would be inserted
    # We want the index i such that cdf[i-1] < u <= cdf[i]
    # searchsorted(cdf, u, side='left') gives the first index where cdf >= u
    # This effectively selects the particle corresponding to the bin u falls into.
    resampled_indices = np.searchsorted(cdf, u, side='left')
    
    # Clip to valid range (just in case of floating point issues)
    resampled_indices = np.clip(resampled_indices, 0, N - 1)
    
    # Resample ensemble
    new_ensemble = ensemble[resampled_indices, :]

    # 4. Spread Restoration / Inflation
    # The resampling step reduces the effective sample size and can reduce spread.
    # We want the ensemble spread at observed locations to be comparable to 
    # the observation error plus the model error.
    # A simple adaptive inflation: compare the ensemble variance at observed 
    # locations to the observation variance.
    
    new_mean = np.mean(new_ensemble, axis=0)
    anomalies = new_ensemble - new_mean
    
    # Compute ensemble variance at observed indices
    # (N-1) denominator for sample variance
    ens_var_obs = np.var(new_ensemble[:, obs_idx], axis=0, ddof=1)
    
    # Target variance: observation variance + a small margin for model error
    # If the ensemble is too tight (under-dispersive), inflate.
    # If it's too wide, we could deflate, but inflation is safer for stability.
    # Let's aim for the ensemble variance to be at least the observation variance.
    # A common heuristic is to inflate until the ensemble variance matches 
    # the observation variance (or a fraction of it).
    
    # Calculate inflation factor alpha
    # We want: alpha^2 * ens_var_obs ~= target_var
    # Let target_var = observation_variance
    # alpha = sqrt(target_var / ens_var_obs)
    
    # To avoid over-inflation when ens_var is already large, we can cap alpha.
    # Also, if ens_var is very small, alpha becomes huge, which is bad.
    # Let's use a smoothed version:
    # alpha = sqrt( max(1, target_var / ens_var_obs) )
    # But we want to apply this to ALL dimensions, not just observed ones?
    # In a particle filter, we usually inflate the anomalies globally to maintain 
    # the covariance structure, or locally. Global is simpler and often effective 
    # for chaotic systems where errors are correlated.
    
    # Average ensemble variance at observed locations
    avg_ens_var_obs = np.mean(ens_var_obs)
    
    if avg_ens_var_obs > 1e-6:
        # Desired ratio of ensemble var to obs var
        # We want the ensemble to be "wide enough" to cover the observation error.
        # Let's target the ensemble variance to be roughly equal to the observation variance.
        # If the ensemble is tighter, inflate.
        ratio = observation_variance / avg_ens_var_obs
        # Cap the ratio to prevent extreme inflation
        # If ratio is < 1, the ensemble is already wider than obs error, so alpha=1
        # If ratio is > 1, we need to inflate.
        alpha = np.sqrt(np.max(1.0, ratio))
        # Cap alpha to avoid blowing up
        alpha = np.min(alpha, 2.0)
    else:
        alpha = 1.0
    
    # Apply inflation to anomalies
    # x_i' = mean + alpha * (x_i - mean)
    new_ensemble = new_mean[None, :] + alpha * anomalies

    # 5. Final Checks and Clipping
    if not np.all(np.isfinite(new_ensemble)):
        new_ensemble = np.nan_to_num(new_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)

    # Time check
    if time.time() > deadline:
        return new_ensemble

    return new_ensemble
