"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf

_DEADLINE_START = time.time()
_TIME_LIMIT = 295.0  # seconds, safety margin before 330s kill


def _norm_cdf(x):
    """Fast standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Structural Change: Cost-Weighted Upper Confidence Bound (UCB).
    
    UCB provides a balance between exploitation (low mu) and exploration (high sigma).
    For minimization, we use an optimistic bound: mu - beta * sigma.
    We maximize the score: - (mu - beta * sigma) / cost.
    This is equivalent to maximizing (beta * sigma - mu) / cost.
    
    Based on cell findings, UCB with beta=2 outperforms EI/KG for this task.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Fallback: simple UCB / cost
        beta = 2.0
        ucb_val = mu - beta * sigma
        score = -ucb_val / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    beta = 2.0
    
    # Optimistic value for minimization
    # We want to find low values.
    # High sigma means potential for lower values.
    ucb_val = mu - beta * sigma
    
    # Score: Higher is better.
    # Low ucb_val is good.
    # So we negate it.
    # Divide by cost to penalize expensive evaluations.
    cost_safe = np.maximum(cost, 1e-9)
    score = -ucb_val / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
