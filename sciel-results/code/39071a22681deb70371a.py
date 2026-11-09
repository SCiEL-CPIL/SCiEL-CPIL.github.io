"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score
    # Baseline: ei / cost
    # Proposed: Softer penalty to avoid over-focusing on the cheapest corners
    # Cost is in [1, 2]. (cost - 1) is in [0, 1].
    # Denominator: 1 + 0.5 * (cost - 1) = 1 + 0.5*cost - 0.5 = 0.5 + 0.5*cost = 0.5*(1+cost)
    # So score = ei / (0.5 * (1 + cost)) = 2 * ei / (1 + cost)
    # At cost=1: 2*ei/2 = ei.
    # At cost=2: 2*ei/3 = 0.66*ei.
    # Baseline at cost=2: 0.5*ei.
    # This makes high-cost points 33% more attractive than baseline, relative to low-cost points.
    
    denom = 1.0 + 0.5 * (cost - 1.0)
    scores = ei / denom
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
