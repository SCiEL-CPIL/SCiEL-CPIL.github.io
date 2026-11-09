"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before the deadline to stop doing heavy work


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) where x is in [0,1]^3.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    # Check deadline: if we're running out of time, return a simple fallback
    if _time_left() <= 0:
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to prefer cheaper evaluations
    # But also consider the magnitude of improvement relative to cost
    # Use a cost-adjusted EI that balances improvement per unit cost
    cost_ei = ei / cost_safe
    
    # Also compute UCB (Upper Confidence Bound) for minimization
    # UCB = mu - kappa * sigma (lower is better)
    # We want high scores for low mu and high sigma
    # Normalize mu and sigma across candidates
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_max = np.max(sigma_safe)
    
    # Normalize to [0, 1]
    if mu_max > mu_min:
        mu_norm = (mu - mu_min) / (mu_max - mu_min)  # 0 = best (lowest), 1 = worst
    else:
        mu_norm = np.zeros_like(mu)
    
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max  # 0 = least uncertain, 1 = most uncertain
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # UCB score: prefer low mu (high mu_norm means bad) and high sigma
    # For minimization, UCB = mu - kappa*sigma, so we want to minimize this
    # Score = -(mu - kappa*sigma) normalized, or equivalently prefer low mu and high sigma
    kappa = 1.5
    ucb_value = mu - kappa * sigma_safe
    ucb_min = np.min(ucb_value)
    ucb_max = np.max(ucb_value)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb_max - ucb_value) / (ucb_max - ucb_min)  # 1 = best (lowest ucb_value)
    else:
        ucb_norm = np.ones_like(mu)
    
    # Cost-adjusted UCB
    cost_ucb = ucb_norm / cost_safe
    
    # Blend cost-aware EI and cost-aware UCB
    # EI favors exploitation (finding better than f_best)
    # UCB favors exploration (uncertain regions)
    alpha = 0.6  # weight for EI
    blended = alpha * cost_ei + (1 - alpha) * cost_ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(blended), blended, 0.0)
    
    # Ensure scores are non-negative for safety
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
