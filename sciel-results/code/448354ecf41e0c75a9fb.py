"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    The key structural change here is applying the cost penalty *differentially*:
    The exploration term (uncertainty bonus) is scaled by 1/cost, while the 
    exploitation term (mean) is not. This creates a non-uniform exploration 
    landscape that prioritizes high-uncertainty, low-cost areas, while still 
    allowing exploitation of low-mean high-cost areas if the mean is sufficiently 
    lower than f_best.
    
    LCB(x) = mu(x) - beta * sigma(x)
    We want to minimize LCB, so we maximize -LCB = -mu + beta * sigma.
    
    To incorporate cost-awareness structurally as directed:
    Score = -mu + beta * (sigma / cost)
    
    This ensures that regions with high cost have their uncertainty bonus 
    diminished, reducing the drive to explore expensive uncertain regions, 
    while the mean term remains invariant to allow finding the minimum 
    even if it lies in a higher-cost region.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost is in range [1, 2] (1 + mean(x scaled to [0,1]))
    # So 1/cost is in range [0.5, 1.0]
    
    # Beta controls the trade-off between exploration and exploitation.
    # Since the exploration term is now divided by cost (>=1), the effective 
    # exploration is reduced. We can use a slightly higher beta to compensate 
    # if needed, but let's start with a standard value.
    beta = 2.0
    
    # Cost-adjusted LCB score (higher is better)
    # -mu: exploit low means
    # beta * sigma / cost: explore high uncertainty, penalized by cost
    scores = -mu + beta * (sigma_safe / cost)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
