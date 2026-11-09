"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
# We keep a module-level timestamp so that repeated calls to acquisition()
# can check elapsed time and bail out early with a safe fallback.
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
        # Fallback: return negative mu (exploit best) divided by cost
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to prefer cheaper evaluations
    cost_ei = ei / cost_safe
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # UCB = mu - kappa * sigma (lower is better)
    # We want high scores for low mu and high sigma
    # Normalize UCB values across candidates
    kappa = 1.5
    ucb_value = mu - kappa * sigma_safe
    ucb_min = np.min(ucb_value)
    ucb_max = np.max(ucb_value)
    
    if ucb_max > ucb_min:
        # Normalize so that best (lowest ucb_value) gets 1, worst gets 0
        ucb_norm = (ucb_max - ucb_value) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(mu)
    
    # Cost-adjusted UCB
    cost_ucb = ucb_norm / cost_safe
    
    # Blend cost-aware EI and cost-aware UCB
    # EI favors exploitation (finding better than f_best)
    # UCB favors exploration (uncertain regions)
    # The current program (score 1.4741) uses a blend of normalized EI and PI
    # with alpha=0.7. Here we blend EI and UCB which is a more standard
    # exploration-exploitation tradeoff.
    # We use a moderate weight on EI to balance exploration and exploitation.
    alpha = 0.7  # weight for EI
    blended = alpha * cost_ei + (1 - alpha) * cost_ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(blended), blended, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
