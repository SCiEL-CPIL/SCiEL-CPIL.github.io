import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill
_total_evals = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses pure Expected Improvement (EI) for minimization.
    This avoids the exploration pitfalls of UCB in highly oscillatory functions
    while maintaining a global view of improvement potential.
    """
    global _total_evals
    _total_evals += 1
    
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Calculate EI
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei_score = np.where(np.isfinite(ei_score), ei_score, 0.0)
    
    return ei_score
