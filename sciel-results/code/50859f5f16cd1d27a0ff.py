"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill

def _within_deadline():
    return (time.time() - _start_time) < (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """Cost-Adjusted Lower Confidence Bound with dynamic kappa."""
    if not _within_deadline():
        # Fallback: pure exploitation (pick lowest mu / cost)
        return -mu / (cost + 1e-9)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Dynamic kappa: increases as cost budget is consumed
    # cost represents the cumulative cost so far; budget is 30
    # When cost is low, explore more; when cost is high, exploit
    total_cost = np.mean(cost)
    # Normalize cost to [0, 1] assuming budget ~30
    cost_fraction = np.clip(total_cost / 30.0, 0.0, 1.0)
    
    # kappa starts at ~1.0 (exploration) and decreases to ~0.1 (exploitation)
    # Actually for minimization LCB: we want to minimize mu - kappa*sigma
    # So higher kappa = more exploration
    # As we get more expensive, we should be more certain -> less exploration
    kappa = 1.5 * (1.0 - cost_fraction) + 0.2
    
    # LCB for minimization: mu - kappa * sigma (lower is better)
    lcb = mu - kappa * sigma_safe
    
    # Cost-adjusted score: we want to maximize (improvement per unit cost)
    # Since we're minimizing, lower LCB is better
    # Score = -LCB / cost (negate because higher score = evaluate next)
    score = -lcb / (cost + eps)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
