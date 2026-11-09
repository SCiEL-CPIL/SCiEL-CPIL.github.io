"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) per unit cost.
    # PI = P(f(x) < f_best)
    # This decouples the "discovery" probability from the "value" magnitude,
    # which can be more robust in flat or multimodal landscapes.
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Cost-aware score: maximize probability of improvement per unit cost
    scores = pi / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
