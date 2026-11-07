"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with a small penalty for high-cost points
    # that are already close to the current best (to encourage exploration).
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Penalize expensive points that are already good (to explore more)
    # and reward cheap points that are promising.
    # cost is in range [1, 2] for this task (1 + mean(x) where x in [0,1]^3)
    # Normalize cost to [0, 1] where 0 is cheapest (cost=1) and 1 is most expensive (cost=2)
    cost_norm = (cost - 1.0) / 1.0  # Now in [0, 1]
    
    # Use a weighted combination: primarily EI per cost, but with a small exploration bonus
    # for cheap points that have decent EI
    base_score = ei / (cost + 1e-9)
    
    # Add a small exploration term: reward cheap points with moderate EI
    # This helps break out of local optima by exploring cheaper regions
    exploration_bonus = 0.1 * (1.0 - cost_norm) * (ei > 0).astype(float) * np.exp(-ei / (np.max(ei) + 1e-9))
    
    return base_score + exploration_bonus
# EVOLVE-BLOCK-END
