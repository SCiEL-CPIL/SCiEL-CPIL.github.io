import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Leave 50s for cleanup/other overhead
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    Combines UCB-style exploration with EI-style exploitation using a smooth
    multiplicative cost discount that preserves relative score ordering while
    uniformly penalizing expensive evaluations.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score: exploit current best
        return -mu.copy()

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB-style score: -mu + sqrt(beta) * sigma
    # beta decays with call count to shift from exploration to exploitation
    beta = 2.0 / (1.0 + 0.05 * _CALL_COUNT)
    ucb_score = -mu + np.sqrt(beta) * sigma

    # Cost-Effective EI: EI per unit cost
    # This naturally prefers cheaper points with similar expected improvement
    cei = ei / np.maximum(cost, 1e-9)

    # Weighted combination: early on UCB dominates, later EI/CEI dominates
    w_ucb = 1.0 / (1.0 + 0.1 * _CALL_COUNT)
    w_ei = 1.0 - w_ucb

    # Base score (higher is better)
    # Use CEI for the EI component to incorporate cost directly
    base_score = w_ucb * ucb_score + w_ei * cei

    # Smooth multiplicative cost discount
    # Cost is in [1.0, 2.0]. We want discount ~1.0 at cost=1.0, ~0.8 at cost=2.0
    # Use an exponential decay: discount = exp(-alpha * (cost - 1.0))
    # At cost=1.0: discount=1.0; at cost=2.0: discount=exp(-alpha)
    # Choose alpha so that exp(-alpha) ~ 0.8 => alpha ~ 0.223
    # But we want this to be less aggressive later in optimization
    alpha = 0.25 / (1.0 + 0.02 * _CALL_COUNT)
    cost_discount = np.exp(-alpha * np.maximum(cost - 1.0, 0.0))

    # Apply smooth discount to base score
    score = base_score * cost_discount

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
