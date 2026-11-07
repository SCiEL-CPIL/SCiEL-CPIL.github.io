"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    This approach samples a function value for each candidate from the 
    Gaussian Process posterior distribution. The candidate with the lowest 
    sampled value is selected for evaluation. To align with the evaluator's 
    requirement that 'higher score means evaluate next', we return the negative 
    of the sampled values.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best (lowest) observed function value so far. (Not used in calculation, 
        as the posterior already reflects all data).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher means evaluate next.
    """
    # Safety deadline guard, though vectorized ops are fast
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    # Standard deviation should be >= 0, but protect against tiny negatives or zeros
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random numbers
    # Using np.random.normal is fast for 2000 elements
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute posterior samples: y ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # We are minimizing the objective.
    # The best candidate is the one with the lowest predicted value (sample).
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (unlikely to be hit, but good practice)
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Fallback to a simple deterministic UCB if we are somehow slow
        # (This is just a safety net, TS is O(N))
        beta = 2.0
        scores = -mu + beta * sigma_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
