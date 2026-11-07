"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combines GP-LCB with cost normalization.
    
    Uses an adaptive beta that balances exploration/exploitation,
    normalized by evaluation cost to prefer cheaper candidates.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute LCB: mu - beta * sigma
    # Use an adaptive beta based on the spread of sigma values
    # Higher beta when uncertainty is high relative to f_best gap
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Base beta around 2.0, adjusted by the relative uncertainty
    # When sigma is large relative to (f_best - mu), explore more
    improvement = f_best - mu
    # Use a soft scaling: beta increases when uncertainty dominates
    # beta = 2.0 + 1.0 * (sigma_mean / (abs(f_best) + 1e-9))
    # But keep it bounded
    beta = 2.0 + 0.5 * (sigma_mean / (abs(f_best) + sigma_mean + 1e-9))
    beta = min(beta, 4.0)  # Cap beta to avoid excessive exploration
    
    lcb = mu - beta * sigma_safe
    
    # Normalize by cost: prefer cheaper evaluations
    # cost is in range [1, 2] (1 + mean of x scaled to [0,1])
    # Use inverse cost weighting
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combine LCB with cost normalization
    # score = lcb / cost  (higher is better since we want low f values)
    # But lcb can be negative, so we need to be careful
    # Actually, for minimization, lower lcb is better, so we want to MAXIMIZE -lcb/cost
    # Or equivalently, we can think of it as: we want candidates with low mu and low sigma
    # that are also cheap.
    
    # Let's use: score = -lcb / cost_safe = (beta * sigma - mu) / cost
    # This rewards low mu (exploitation) and high sigma (exploration), penalized by cost
    score = (-lcb) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
