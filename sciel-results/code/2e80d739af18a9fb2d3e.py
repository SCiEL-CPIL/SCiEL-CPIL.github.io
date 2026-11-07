"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a cost-aware exploration
    bonus. The key design choices:
    
    1. Use EI as the primary exploitation signal.
    2. Divide by cost to get improvement per unit cost (cost-awareness).
    3. Add a small exploration bonus based on sigma to encourage exploring
       uncertain regions, but dampened by cost.
    4. Clip z-values to avoid numerical issues.
    5. Include a deadline guard to return valid results before timeout.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback: prefer low mu (exploitation) with cost awareness
        return (f_best - mu) / (cost + 1e-9)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow in norm.pdf/cdf
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: improvement per unit cost
    cost_ei = ei / cost
    
    # Exploration bonus: encourage exploring high-uncertainty regions
    # Dampened by cost to avoid expensive exploration
    # Use a small coefficient to balance exploitation and exploration
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine: primary signal is cost-aware EI, secondary is exploration
    score = cost_ei + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
