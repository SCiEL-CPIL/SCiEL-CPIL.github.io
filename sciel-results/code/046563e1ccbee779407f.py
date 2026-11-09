"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds of safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    This deterministic approach balances exploitation (EI) with exploration (UCB).
    The EI term focuses on regions likely to improve the best observed value,
    while the UCB term encourages visiting regions with high posterior uncertainty.
    This combination has shown superior performance for multimodal functions
    like Goldstein-Price compared to pure Thompson Sampling or pure EI.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Fallback to simple UCB if time is running out
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 1.5 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score: (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above the mean
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # UCB component: -mu + beta * sigma
    # Beta controls the exploration-exploitation tradeoff
    # A value of 1.5 is a standard choice that balances both well
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB
    # We use equal weighting as both terms are in objective units and
    # provide complementary signals.
    scores = 0.5 * ei + 0.5 * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
