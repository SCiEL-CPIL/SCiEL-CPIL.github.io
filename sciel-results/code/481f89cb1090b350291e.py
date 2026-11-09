"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _quantile_lcb(mu, sigma, f_best, cost, q=0.05, beta=2.0):
    """Quantile-based lower confidence bound with non-linear cost discount."""
    # Dynamic quantile: use 5th percentile of posterior (aggressive exploration)
    # z-value for 5th percentile is approximately -1.645
    z_q = -1.645
    # Lower quantile of the predictive distribution
    lower_quantile = mu + z_q * sigma
    
    # Non-linear cost penalty using sigmoidal discount
    # cost ranges from 1 to 2 (since mean(x) is in [0,1], cost = 1 + mean(x))
    # We want to heavily penalize high-cost candidates
    # Normalize cost to [0, 1]
    cost_norm = (cost - 1.0) / 1.0  # cost is in [1, 2]
    
    # Sigmoidal discount: aggressive pruning of high-cost candidates
    # As cost_norm increases, discount decreases rapidly
    discount = 1.0 / (1.0 + np.exp(15.0 * (cost_norm - 0.3)))
    
    # LCB score: higher is better (we want low values, so negative of lower quantile)
    # Apply cost discount
    score = -lower_quantile * discount
    
    return score


def _ei_score(mu, sigma, f_best, cost):
    """Expected improvement per unit cost."""
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / cost


def _hybrid_score(mu, sigma, f_best, cost):
    """Hybrid of quantile-LCB and cost-aware EI."""
    # Use quantile-LCB for exploration, EI for exploitation
    qlcb = _quantile_lcb(mu, sigma, f_best, cost, q=0.05, beta=2.0)
    ei = _ei_score(mu, sigma, f_best, cost)
    
    # Combine with adaptive weighting
    # When uncertainty is high, rely more on quantile-LCB
    # When uncertainty is low, rely more on EI
    sigma_mean = np.mean(sigma)
    if sigma_mean > 0.5:
        weight_qlcb = 0.7
    elif sigma_mean > 0.2:
        weight_qlcb = 0.4
    else:
        weight_qlcb = 0.2
    
    weight_ei = 1.0 - weight_qlcb
    
    # Normalize scores to comparable scales
    qlcb_norm = (qlcb - np.min(qlcb)) / (np.max(qlcb) - np.min(qlcb) + 1e-9)
    ei_norm = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)
    
    return weight_qlcb * qlcb_norm + weight_ei * ei_norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function using quantile-based LCB hybrid."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    try:
        # Use hybrid score
        scores = _hybrid_score(mu, sigma, f_best, cost)
        
        # Ensure finite values
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        # Check time
        if time.time() > deadline:
            return scores
        
        return scores
        
    except Exception:
        # Fallback to simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        scores = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
# EVOLVE-BLOCK-END
