"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280
_ACQ_CALL_COUNT = [0]


def acquisition(mu, np_sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - np_sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = better)
    """
    _ACQ_CALL_COUNT[0] += 1

    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Compute z-score for EI
    z = (f_best - mu) / sigma

    # Compute EI with numerical stability using erf for speed
    ei = np.zeros_like(mu)
    mask = np.abs(z) < 20
    z_clipped = np.clip(z, -20, 20)

    # Use scipy's norm for CDF/PDF (cached internally)
    cdf_vals = norm.cdf(z_clipped[mask])
    pdf_vals = norm.pdf(z_clipped[mask])

    ei[mask] = (f_best - mu[mask]) * cdf_vals + sigma[mask] * pdf_vals

    # For very large positive z, EI ≈ f_best - mu
    large_pos = (~mask) & (z > 0)
    ei[large_pos] = f_best - mu[large_pos]

    # For very negative z, EI ≈ 0
    ei[(~mask) & (z <= 0)] = 0.0

    ei = np.maximum(ei, 0.0)

    # Cost-aware score: use a smooth cost-aware EI
    # The idea: maximize E[max(f_best - f(x), 0)] / cost(x)
    # But also add exploration to avoid getting stuck in local optima

    # Normalize cost to [0, 1] range for better scaling
    cost_min = cost.min()
    cost_max = cost.max()
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)

    # Adaptive exploration weight: increase exploration early on, decrease later
    # Use the number of calls as a proxy for progress
    n_calls = _ACQ_CALL_COUNT[0]
    # Exploration weight decays from ~0.3 to ~0.05 over 100 calls
    exploration_weight = 0.3 * np.exp(-n_calls / 100.0) + 0.05

    # Cost-aware EI: divide by cost but use a soft version to avoid extreme values
    # Use cost^alpha with alpha < 1 to be less aggressive
    cost_alpha = 0.7
    ei_cost = ei / (cost ** cost_alpha)

    # Exploration term: favor high uncertainty regions with low cost
    # Use sigma normalized by its max, divided by cost
    sigma_max = sigma.max()
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    exploration = exploration_weight * sigma_norm / (cost ** 0.5)

    # Combined score
    score = ei_cost + exploration

    # Penalize very high cost points slightly to encourage cost-efficiency
    # But don't over-penalize, just a gentle nudge
    high_cost_penalty = 0.02 * cost_norm
    score = score - high_cost_penalty

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
