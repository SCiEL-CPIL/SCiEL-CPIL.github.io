"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 300s as safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    Combines normalized expected improvement with a cost-aware exploration
    term based on posterior uncertainty relative to the scale of f_best.
    
    Uses a different design: instead of raw EI/UCB, we normalize both terms
    by their typical scales to create a dimensionless score that balances
    exploitation and exploration more robustly.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Safe fallback: simple cost-normalized EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        ei_cost = ei / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        return score
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(0, f_best - f(x))]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI by the typical improvement scale to make it dimensionless
    # Use the spread of mu as a natural scale
    mu_std = np.std(mu)
    if mu_std < 1e-12:
        mu_std = max(abs(f_best), 1e-6)
    
    # Normalized EI
    ei_norm = ei / (mu_std + 1e-9)
    
    # Exploration term: posterior uncertainty normalized by cost
    # This encourages visiting regions with high uncertainty that are cheap to evaluate
    # Use a different structure: uncertainty relative to the range of sigma
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe)
    if sigma_range < 1e-12:
        sigma_range = max(np.mean(sigma_safe), 1e-6)
    
    # Uncertainty score: how much does this point's sigma exceed the median?
    sigma_median = np.median(sigma_safe)
    uncertainty_bonus = np.maximum(sigma_safe - sigma_median, 0.0) / (sigma_range + 1e-9)
    
    # Cost-awareness: divide by cost to prefer cheap evaluations
    # But also add a small constant to avoid division by very small costs
    cost_factor = 1.0 / cost_safe
    
    # Balance exploitation and exploration
    # Weight EI heavily but add a meaningful exploration component
    # The exploration weight adapts based on how well we've found the minimum
    mu_min = np.min(mu)
    gap = f_best - mu_min
    if gap < 1e-12:
        explore_weight = 0.1
    else:
        # Normalize gap by the scale of the problem
        explore_weight = min(1.0, gap / (mu_std + 1e-9))
        explore_weight = 0.1 + 0.9 * explore_weight
    
    score = ei_norm * cost_factor + explore_weight * uncertainty_bonus * cost_factor
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
