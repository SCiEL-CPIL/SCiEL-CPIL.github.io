"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Uses a structure-aware approach:
    1. Add a large bonus to candidates near the known global minimum region
    2. Use cost-adjusted Expected Improvement for exploitation
    3. Use cost-adjusted LCB for exploration
    
    The Styblinski-Tang 2D function has its global minimum at x* = (-2.903534, -2.903534)
    with value -78.3323. In the scaled [0,1]^2 space, this corresponds to approximately
    (0.2093, 0.2093) since scaling is (x + 5) / 10.
    
    However, the problem states that each instance f_k(x) = f(x - s_k) moves the minimizer
    to a uniform point in the inner 80% of each range. So the actual minimum location varies
    per instance. We cannot rely on a fixed location.
    
    Instead, we use:
    1. Cost-adjusted Expected Improvement (primary exploitation signal)
    2. Cost-adjusted Lower Confidence Bound (exploration)
    3. A small bonus for high uncertainty (sigma) to encourage exploration early
    
    The key insight: since the minimizer location is unknown and varies per instance,
    we should rely on the GP's predictive distribution (mu, sigma) rather than
    hardcoded locations.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound for minimization (we want low predicted values)
    # LCB = mu - beta * sigma; lower LCB means potentially better point
    # For acquisition score (higher = better), we use -LCB or equivalently prefer low mu + high sigma
    beta = 0.8
    lcb_score = -mu + beta * sigma_safe
    
    # Cost adjustment: divide by cost to favor cheaper evaluations
    # cost is in the range [1, 2] (1 + mean(x_scaled) where x_scaled in [0,1])
    # So cost is between 1 and 2
    
    # Normalize EI by cost
    ei_adjusted = ei / (cost + 1e-12)
    
    # Normalize LCB score by cost
    lcb_adjusted = lcb_score / (cost + 1e-12)
    
    # Combine: weight EI more heavily as it's a direct improvement estimate
    # Add a small exploration bonus based on sigma/cost
    gamma = 0.5
    exploration_bonus = gamma * sigma_safe / (cost + 1e-12)
    
    # Final score: blend of cost-adjusted EI, cost-adjusted LCB, and exploration
    # EI is in objective units, LCB is also in objective units
    # Scale EI to be comparable: EI can be large early on
    score = 0.7 * ei_adjusted + 0.3 * lcb_adjusted + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
