import numpy as np
import time

# Track time to ensure we don't exceed the 330s limit
_start_time = None
_DEADLINE = 315.0  # seconds, safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. A tuned exploration bonus
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means evaluate next.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline with safety margin
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple fallback if we're over deadline
        return -mu + 0.5 * sigma
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to balance exploitation and exploration
    # Use a moderate beta to encourage exploration in early stages
    beta = 0.3
    scores = ei + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
