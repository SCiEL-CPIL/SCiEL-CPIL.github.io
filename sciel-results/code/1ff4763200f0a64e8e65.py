"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Lower Confidence Bound (LCB), with adaptive exploration parameter.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just use -mu / cost as a simple heuristic
        return -mu / (cost + 1e-9)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute EI (Expected Improvement)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute LCB with adaptive beta
    # beta scales with the relative uncertainty: higher when sigma is large relative to improvement
    # This encourages exploration in high-uncertainty regions
    # Use a dynamic beta that balances exploitation and exploration
    # beta = 2 * log(1 + sigma / (sigma.mean() + 1e-9)) gives more exploration to uncertain points
    sigma_mean = np.mean(sigma)
    beta = 2.0 * np.log(1.0 + sigma / (sigma_mean + 1e-9))
    
    # LCB score: lower predicted value with uncertainty penalty
    lcb = mu - beta * sigma
    
    # Combine EI and LCB
    # Normalize both to comparable scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    
    ei_range = ei_max - ei_min + 1e-9
    lcb_range = lcb_max - lcb_min + 1e-9
    
    ei_norm = (ei - ei_min) / ei_range
    lcb_norm = (lcb_min - lcb) / lcb_range  # Invert so higher is better
    
    # Weighted combination: 0.6 * EI + 0.4 * LCB
    combined = 0.6 * ei_norm + 0.4 * lcb_norm
    
    # Divide by cost to get cost-aware score
    scores = combined / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
