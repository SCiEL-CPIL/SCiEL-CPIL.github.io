"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for the deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 330s limit

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Probability of Improvement (PI) normalized by cost.
    PI is more robust to multimodality than EI in low-noise settings.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement (PI)
    # PI = Phi(z)
    # This measures the likelihood that mu < f_best
    pi = norm.cdf(z)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Base score: PI normalized by cost
    # Higher PI means higher probability of finding a better point.
    # Dividing by cost penalizes expensive points.
    base_score = pi / cost
    
    # Add a small exploration bonus proportional to sigma / cost
    # This encourages visiting unexplored areas, which is crucial for Levy 2D
    # to avoid getting stuck in local minima.
    # We use a smaller coefficient than before to keep PI dominant.
    exploration_bonus = 0.05 * sigma / cost
    
    scores = base_score + exploration_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If we're close to the deadline, return a simpler score to save time
    if _elapsed() > _DEADLINE_SECONDS:
        # Fall back to just PI/cost without the bonus computation
        scores = np.nan_to_num(base_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
