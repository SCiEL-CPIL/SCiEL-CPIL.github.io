"""Cost-aware acquisition for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_MARGIN_S = 5.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _deadline_reached():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN_S)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe finite score that picks the lowest-cost candidate
    if _deadline_reached():
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e30, dtype=np.float64)
        scores[idx] = 0.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Cost-aware acquisition combining EHI (exploitation) with cost-aware UCB (exploration)
    # 
    # EHI: smoothed expected improvement, damped by uncertainty
    # UCB: optimistic upper confidence bound, damped by cost
    #
    # The key insight: cheap candidates with high uncertainty should be explored,
    # while expensive candidates should only be evaluated if they promise high improvement.

    # --- EHI component (exploitation) ---
    c = 0.3  # scale parameter for smoothing (smaller = more aggressive)
    improvement = f_best - mu
    # Use a softer damping than pure Gaussian to preserve some signal at moderate sigma
    gaussian_damping = np.exp(-sigma**2 / (2 * c**2))
    ehi = improvement * gaussian_damping / c

    # --- Cost-aware UCB component (exploration) ---
    # UCB = mu + beta * sigma, but we want to explore where sigma is high AND cost is low
    # Use a cost-modulated exploration bonus
    beta = 1.5  # exploration weight
    cost_normalized = cost - 1.0  # in [0, 1]
    # Exploration bonus is reduced for expensive points
    cost_damp_explore = 1.0 / (1.0 + 1.5 * cost_normalized)
    ucb_bonus = beta * sigma * cost_damp_explore

    # --- Combine ---
    # EHI gives the exploitation signal, UCB bonus adds exploration for cheap uncertain points
    # The EHI is already cost-penalized implicitly (high-cost regions have been explored less,
    # so sigma is higher there, but we don't want to explore expensive uncertain regions)
    ehi_cost_penalty = 1.0 + 1.0 * cost_normalized  # mild cost penalty on exploitation
    score = ehi / ehi_cost_penalty + ucb_bonus

    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e30)

    # Clip to avoid extreme values
    score = np.clip(score, -1e10, 1e10)

    return score
# EVOLVE-BLOCK-END
