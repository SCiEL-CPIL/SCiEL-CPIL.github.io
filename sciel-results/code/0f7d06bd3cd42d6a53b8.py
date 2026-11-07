"""Cost-aware acquisition for Bayesian optimization of Rastrigin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement (PoI) normalized by cost.
    
    PoI is more robust for multimodal functions like Rastrigin because it focuses
    on the likelihood of finding a better point rather than the magnitude of improvement.
    For minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    We normalize by cost to prefer cheaper evaluations, and add a small exploration
    bonus for high-uncertainty regions to handle multimodality.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Probability of Improvement for minimization:
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    # Clip to [0, 1] for numerical safety
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-aware score: PoI per unit cost
    score = poi / cost
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in multimodal landscapes
    # The bonus is scaled by sigma and decays with cost
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine: PoI/cost + exploration bonus
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
