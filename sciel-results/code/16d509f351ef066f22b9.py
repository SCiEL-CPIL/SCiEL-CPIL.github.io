"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _elapsed_ok():
    return (time.time() - _start_time) < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if not _elapsed_ok():
        # Emergency: return simple UCB scores
        return np.maximum(f_best - mu + 2.0 * sigma, 0.0) / np.maximum(cost, 1e-6)

    n = len(mu)
    # Robust sigma floor
    sigma_safe = np.maximum(sigma, 1e-8)

    # --- Compute base acquisition scores (EI + UCB blend) ---
    # EI for minimization: max(f_best - mu, 0) + correction
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    from scipy.stats import norm
    ei = sigma_safe * (z * norm.pdf(z) + norm.cdf(z))
    ei = np.maximum(ei, 0.0)

    # UCB for minimization
    ucb = f_best - mu + 2.5 * sigma_safe

    # Blend: 0.6 * EI + 0.4 * UCB
    base = 0.6 * ei + 0.4 * ucb

    # --- Cost normalization ---
    # Normalize cost to [0, 1] range for this batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros(n)

    # Cost penalty: higher cost -> lower score
    # Use a soft penalty: divide by (1 + alpha * cost_norm)
    alpha = 1.5
    cost_penalty = 1.0 / (1.0 + alpha * cost_norm)

    scores = base * cost_penalty

    # --- Batch-aware diversity penalty (greedy redundancy reduction) ---
    # We don't have coordinates, but we can use the sigma field as a proxy
    # for spatial information. Points with very low sigma relative to neighbors
    # are likely redundant. Instead, we use a simple strategy:
    # Penalize points whose base score is very close to the top few scores
    # (they're likely in the same region).

    # Sort by base score descending
    sorted_idx = np.argsort(-base)
    top_k = min(50, n)

    # For points in the top region, apply a mild diversity penalty based on
    # how many other points have similar base scores (proxy for same region)
    # This is a heuristic since we lack coordinates.
    # Use sigma as a proxy: low sigma = confident = likely already explored
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-12:
        sigma_ratio = sigma_safe / sigma_mean
        # Points with very low sigma (confident predictions) get slightly penalized
        # to encourage exploration of uncertain regions
        diversity_bonus = 0.3 * np.log1p(sigma_ratio)
    else:
        diversity_bonus = np.zeros(n)

    scores = scores * (1.0 + diversity_bonus)

    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)

    if not _elapsed_ok():
        return scores

    return scores
# EVOLVE-BLOCK-END
