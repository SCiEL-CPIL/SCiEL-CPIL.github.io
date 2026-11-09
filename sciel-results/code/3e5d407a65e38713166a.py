"""Cost-aware acquisition for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_DEADLINE = None
_TIME_LIMIT = 325.0  # seconds, safety margin before 330s kill

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Strategy: Use a cost-aware variant of Expected Improvement that balances
    exploration against the cumulative cost budget. We normalize EI by the
    ratio of current point cost to mean cost, creating a dynamic efficiency
    metric. Additionally, we add a small UCB-like exploration term to avoid
    getting stuck in local exploitation.
    """
    n = len(mu)
    
    # Ensure finite inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    mu_clean = np.where(np.isfinite(mu), mu, f_best)
    sigma_clean = np.where(np.isfinite(sigma_safe), sigma_safe, 1.0)
    cost_clean = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute Expected Improvement
    improvement = f_best - mu_clean
    z = improvement / sigma_clean
    
    # Handle numerical stability for very large/small z
    z = np.clip(z, -50, 50)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma_clean * pdf
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization: divide by cost to get "improvement per unit cost"
    # But also consider the cumulative budget perspective
    mean_cost = np.mean(cost_clean)
    cost_ratio = cost_clean / (mean_cost + 1e-9)
    
    # Primary score: EI per unit cost (normalized)
    score_ei = ei / cost_ratio
    
    # Exploration term: UCB-style bonus for high uncertainty
    # This helps escape local optima, especially important for multi-modal functions
    uncertainty_bonus = sigma_clean / (sigma_clean + 0.5)
    exploration_weight = 0.1  # Small weight to not dominate exploitation
    
    score_ucb = uncertainty_bonus
    
    # Combined score
    score = score_ei + exploration_weight * score_ucb
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time check: if we're running out of time, return a simple deterministic score
    if _time_remaining() < 5.0:
        # Fallback: just use EI/cost to be safe and fast
        fallback = ei / cost_clean
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
