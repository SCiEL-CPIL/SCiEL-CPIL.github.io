"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 300s as safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Structural Change: Use Cost-normalized Probability of Improvement (PoI).
    
    Instead of Expected Improvement (which integrates the magnitude of improvement),
    we use the Probability of Improvement (which measures the likelihood of beating f_best).
    
    Score(x) = P(f(x) < f_best) / cost(x)
             = Phi((f_best - mu(x)) / sigma(x)) / cost(x)
             
    This is a different acquisition family. It tends to be more robust in early stages
    and focuses on "safe" improvements per unit cost. It avoids the complexity of EI
    and the heuristic tuning of UCB weights.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Fallback: simple PoI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    # If f_best < mu, z is negative, PoI is small.
    # If f_best > mu, z is positive, PoI is large.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using scipy.stats.norm.cdf for precision
    poi = norm.cdf(z)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-9)
    score = poi / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
