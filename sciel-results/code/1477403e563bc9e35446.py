"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: 330s wall-clock limit, use 300s as safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with adaptive EI/UCB mixing.
    
    Uses Expected Improvement normalized by cost, blended with a UCB term.
    The blend weight adapts based on how close f_best is to the predicted
    minimum: when we're stuck (f_best >> min(mu)), explore more; when
    making progress, exploit more.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * np.maximum(0, z)
        ei_cost = ei / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        return score
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard EI computation (always non-negative)
    ei = (f_best - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Adaptive exploration weight:
    # When f_best is much larger than min(mu), we haven't found the minimum yet
    # -> explore more. When f_best is close to min(mu), exploit.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-12:
        exploration_weight = 0.5
    else:
        # f_best_norm: 0 means f_best == mu_min (great), 1 means f_best == mu_max (stuck)
        f_best_norm = (f_best - mu_min) / (mu_range + 1e-12)
        # Clamp to [0, 1]
        f_best_norm = np.clip(f_best_norm, 0.0, 1.0)
        # Sigmoid: low f_best_norm -> low exploration (exploit), high -> high exploration
        # Use steeper transition for clearer phase shift
        exploration_weight = 1.0 / (1.0 + np.exp(-6.0 * (f_best_norm - 0.3)))
    
    # UCB term: use a beta that scales with the problem scale
    # For levy 2d, typical values range from 0 to ~1000+
    # Scale beta so UCB is meaningful relative to EI
    mu_scale = max(np.mean(np.abs(mu)), 1e-6)
    beta = 1.5 * np.sqrt(mu_scale) / (np.mean(sigma_safe) + 1e-6)
    beta = np.clip(beta, 1.0, 5.0)
    
    ucb_cost = (beta * sigma_safe) / cost_safe
    
    # Combine with adaptive weighting
    score = (1.0 - exploration_weight) * ei_cost + exploration_weight * ucb_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score


def _norm_cdf(x):
    """Fast approximation of standard normal CDF."""
    # Use the error function approximation
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast computation of standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _erf(x):
    """Approximation of the error function using Abramowitz and Stegun."""
    # Coefficients
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = np.sign(x)
    x_abs = np.abs(x)
    
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
    
    return sign * y
# EVOLVE-BLOCK-END
