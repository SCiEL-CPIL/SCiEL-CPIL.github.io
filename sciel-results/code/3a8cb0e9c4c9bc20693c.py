"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_eval_count = 5  # Start counting after the initial 5 random points

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Thompson Sampling with multiple samples to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _eval_count
    
    if _start_time is None:
        _start_time = time.time()
        
    # Increment the evaluation counter
    _eval_count += 1
    
    # Safety margin: If the total time exceeds 290s, we return a simplified score
    # to ensure we don't risk a timeout in the evaluator's loop.
    if time.time() - _start_time > 290:
        # Fall back to a pure exploitation strategy (minimize mu)
        # to finish quickly.
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive to avoid division by zero or invalid values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling
    # We draw K samples from the posterior for each candidate and take the minimum.
    # This approximates the "best" value this point could achieve, favoring points
    # with low mean or high variance (if the sample hits low).
    
    # Number of samples per candidate. 
    # K=5 is a good balance between computational cost and robustness for N=2000.
    K = 5
    
    # Generate random standard normal samples
    # Shape: (K, 2000)
    Z = np.random.randn(K, len(mu))
    
    # Compute samples: Y_k = mu + sigma * Z_k
    # mu and sigma are (2000,), so we expand dims to broadcast
    # Y shape: (K, 2000)
    Y = mu[None, :] + sigma_safe[None, :] * Z
    
    # For each candidate i, we want the minimum of the K samples
    # This represents the "optimistic" best value for that candidate
    min_samples = np.min(Y, axis=0)
    
    # The acquisition score should be higher for better candidates.
    # Better candidates have lower function values.
    # So we negate the min_samples to make lower values higher scores.
    scores = -min_samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
