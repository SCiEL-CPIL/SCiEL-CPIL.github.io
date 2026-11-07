# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a Thompson Sampling variant: sample from the posterior predictive
    distribution to create a stochastic acquisition score. This naturally balances
    exploration and exploitation without fixed hyperparameters.
    """
    start_time = time.time()
    deadline = 295.0  # 300s limit, keep 5s margin for safety

    # Convert to arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Sanitize inputs
    # Ensure sigma is strictly positive to prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Replace non-finite mu with f_best (worst case assumption for unknowns)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Replace non-finite sigma with a reasonable default
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-6)
    
    # Thompson Sampling: sample from the posterior predictive distribution
    # For minimization, we want to select points with low predicted values
    # Sample: y ~ N(mu, sigma^2)
    # Since we're minimizing, lower samples are better
    # We use a single sample from the Gaussian posterior
    
    # Use a fixed random seed for reproducibility within the acquisition call
    # Actually, we should use a fresh random draw each time to get stochasticity
    # But we need to be careful about the random state
    
    # Draw a random sample from the posterior predictive
    # y_sample = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.randn(len(mu))
    
    # The sampled value represents our "belief" about the function value
    # For minimization, we want to maximize the probability of improvement
    # or minimize the sampled value
    
    # Thompson Sampling for minimization:
    # We sample from the posterior and select the point with the lowest sample
    # This is equivalent to maximizing -sample
    
    # However, a pure Thompson Sampling approach might be too exploratory
    # Let's use a variant that incorporates the current best
    
    # Alternative: use the sampled value directly
    # Lower sampled values are preferred for minimization
    # So we return -sampled_value as the score (higher is better)
    
    # But this might be too noisy. Let's use a combination:
    # Score = -sampled_value + small_exploration_bonus
    
    # Actually, let me think about this more carefully.
    # In Thompson Sampling for minimization:
    # 1. For each candidate x, sample f(x) ~ N(mu(x), sigma^2(x))
    # 2. Select x* = argmin f(x)
    
    # So the acquisition score should be the negative of the sampled value
    # (since we want to maximize the score to select the point)
    
    sampled_values = mu + sigma_safe * z
    
    # For minimization, we want to select points with low sampled values
    # So the score is -sampled_values
    score = -sampled_values
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Replace -inf with a very large negative number to avoid issues
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return negative mu (exploitation only) to ensure validity
        fallback_score = -np.where(np.isfinite(mu), mu, f_best)
        fallback_score = np.where(np.isfinite(fallback_score), fallback_score, -1e10)
        return fallback_score
        
    return score
# EVOLVE-BLOCK-END
