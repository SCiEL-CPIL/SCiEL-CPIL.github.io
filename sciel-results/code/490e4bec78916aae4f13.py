# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time from module load.
_t0 = time.time()
_DEADLINE = 325.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP std dev predictions in objective units
        f_best: scalar, lowest observed function value so far
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = better)
    """
    # Time check: if we're close to deadline, return a simple baseline quickly
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Cost-aware acquisition using max(EI, LCB) to avoid interference
    # between exploitation and exploration terms.
    
    # Parameters
    xi = 0.005        # small exploration in EI
    kappa = 1.5       # LCB exploration weight (beta in some conventions)
    cost_power = 1.2  # cost penalty exponent
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement with exploration parameter xi
    z_ei = (f_best - mu - xi) / sigma_safe
    ei = (f_best - xi - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) for exploration
    # LCB = mu - kappa * sigma (lower is better for minimization)
    # Convert to "improvement" scale: higher LCB score = more promising
    lcb_score = -(mu - kappa * sigma_safe)
    
    # Use max to prevent interference: fully exploit or fully explore
    base_score = np.maximum(ei, lcb_score)
    
    # Cost-aware weighting: prefer cheaper evaluations
    # cost is in [1, 2], so normalize to [0, 1] where 0 = cheapest
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-9:
        cost_norm = (cost - cost_min) / cost_range  # [0, 1], lower is cheaper
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost factor: (1 - cost_norm)^power, so cheapest gets ~1, most expensive gets ~0
    cost_factor = (1.0 - cost_norm) ** cost_power
    
    # Final score: base_score weighted by cost factor
    score = base_score * cost_factor
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
