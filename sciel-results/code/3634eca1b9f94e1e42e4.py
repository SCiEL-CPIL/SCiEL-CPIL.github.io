"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware EI with reduced cost sensitivity and exploration term
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a UCB-like exploration term to encourage exploration in the valley
    exploration = 1.0 * sigma
    
    # Reduce the impact of cost to avoid biasing away from high-cost minima
    # Cost is in [1, 2]. Original penalty was 2x. New penalty is 1.2x.
    # denom = 1.0 + 0.2 * (cost - 1.0)
    denom = 1.0 + 0.2 * (cost - 1.0)
    
    return (ei + exploration) / denom
# EVOLVE-BLOCK-END
