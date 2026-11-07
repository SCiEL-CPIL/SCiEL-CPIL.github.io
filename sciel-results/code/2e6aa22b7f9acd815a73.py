# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using UCB + Probability of Improvement.
    
    Structural change: Instead of EI-based scoring with meta-selection,
    we use a two-component acquisition:
      1. UCB (lower confidence bound) / cost  -- exploitation with confidence
      2. P(improve) * sigma / cost            -- uncertainty-driven exploration
    
    This is a different modeling family: rather than expected improvement
    (which can be dominated by high-sigma regions), we separate the
    confidence in improvement from the raw uncertainty, and weight both
    by cost efficiency.
    """
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        scores = (f_best - mu) / np.sqrt(cost) + 0.5 * sigma / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    z = (f_best - mu) / sigma

    # Component 1: UCB-style -- (f_best - mu - beta*sigma) / cost
    # This penalizes both high mean AND high uncertainty, divided by cost.
    # Beta controls exploration-exploitation tradeoff.
    beta = 1.0
    ucb = (f_best - mu - beta * sigma) / cost
    ucb = np.maximum(ucb, 0.0)

    # Component 2: Probability of improvement weighted by uncertainty / cost
    # P(f(x) < f_best) = Phi(z). Multiply by sigma to reward uncertain points
    # that have a reasonable chance of improvement.
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 0.0, 1.0)
    exploration = p_improve * sigma / cost

    # Combine: UCB dominates when confident, exploration fills in when uncertain
    # Weight: 0.7 for UCB, 0.3 for exploration
    w_ucb = 0.7
    w_exp = 0.3
    scores = w_ucb * ucb + w_exp * exploration

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
