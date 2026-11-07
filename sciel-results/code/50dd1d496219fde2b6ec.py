"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard: track when the process started.
# The evaluator kills at 330s wall-clock; we use a safety margin.
_PROCESS_START = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill to bail out safely
_DEADLINE = _PROCESS_START + 330.0 - _DEADLINE_MARGIN


def _time_remaining():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) for minimization.

    Structural Change:
    Instead of adaptive Expected Improvement (EI), we use a Cost-Aware UCB.
    This method maximizes the optimistic potential improvement (based on the
    lower confidence bound of the function value) divided by the cost.
    
    Score = (f_best - (mu - beta * sigma)) / cost
    
    This encourages exploration in high-uncertainty regions (large sigma)
    that could potentially yield values much lower than f_best, while
    penalizing expensive evaluations.
    """
    # Deadline guard: if we're running out of time, fall back to a simple
    # cost-aware EI that is fast to compute.
    if _time_remaining() < 0.5:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Cost-aware UCB
    # beta controls the exploration-exploitation trade-off.
    # A fixed beta is used here to avoid the instability of adaptive heuristics.
    beta = 10.0
    
    # Lower confidence bound for the function value (minimization)
    # We want to find points where the true value f(x) is likely to be low.
    # The lower bound of the confidence interval is mu - beta * sigma.
    # If this lower bound is significantly lower than f_best, it's a promising candidate.
    lcb = mu - beta * sigma
    
    # Potential improvement: how much better could f(x) be than f_best?
    # If lcb < f_best, the improvement is f_best - lcb.
    # If lcb >= f_best, the improvement is 0 (or negative, which we clip).
    improvement = f_best - lcb
    improvement = np.maximum(improvement, 0.0)
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    score = improvement / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
