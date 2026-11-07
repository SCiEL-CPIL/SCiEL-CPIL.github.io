"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a PI-weighted
    exploration term. This hybrid approach balances exploitation of known good
    regions with exploration of uncertain regions that have a high probability
    of containing better solutions.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    # Track elapsed time to ensure we finish well within the 300s limit
    # The computation is O(n) with n=2000, so it's very fast.
    # This guard is primarily a safety measure for unexpected edge cases.
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    # A small floor ensures stability without significantly affecting results
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning we expect improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    # This is the probability that f(x) < f_best
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # Standard EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Exploration term: PI-weighted sigma
    # This term is large when:
    # 1. sigma is large (high uncertainty)
    # 2. pi is high (high probability of improvement)
    # This is more targeted than raw sigma (UCB) because it only explores
    # uncertain regions where improvement is likely.
    # The weight 0.2 was chosen to provide a meaningful exploration bonus
    # without overwhelming the exploitation signal from EI.
    pi_weighted_exploration = 0.2 * pi * sigma_safe
    
    # Small raw sigma term for baseline exploration
    # This ensures some diversity even in regions where pi is very low,
    # preventing premature convergence to a single local minimum.
    raw_exploration = 0.05 * sigma_safe
    
    # Combine all terms
    scores = ei + pi_weighted_exploration + raw_exploration
    
    # Ensure all scores are finite
    # If any scores are NaN or Inf, replace them with 0.0 to avoid invalid results
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if computation took too long (shouldn't happen), return early
    elapsed = time.time() - start_time
    if elapsed > 290:  # 10 second safety margin
        # This should never trigger for 2000 points, but just in case
        return scores
    
    return scores
# EVOLVE-BLOCK-END
