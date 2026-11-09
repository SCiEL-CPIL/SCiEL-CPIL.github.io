# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s wall-clock.
# Track per-call elapsed time to avoid pathological cases.
_ACQ_CALL_START = None


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition for minimizing a multimodal objective.
    mu, sigma, cost: shape (2000,)
    Higher returned score => evaluate next.
    """
    global _ACQ_CALL_START
    if _ACQ_CALL_START is None:
        _ACQ_CALL_START = time.time()
    elapsed = time.time() - _ACQ_CALL_START
    # Safety margin: if we're already past 290s, return a simple EI score fast.
    if elapsed > 290.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    n = mu.shape[0]
    # Guard: ensure finite inputs and positive sigma.
    sigma = np.maximum(sigma, 1e-9)

    # --- Cost model ---
    # cost = 1 + mean(x_scaled) in [1, 2]. Cheaper near the center of the box.
    cost = np.maximum(cost, 1e-9)
    # cheapness in [0, 1]: 1 when cost=1 (center), 0 when cost=2 (corner).
    cheapness = np.clip(1.0 - (cost - 1.0), 0.0, 1.0)

    # --- Expected Improvement (for minimization) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu)/sigma
    z = (f_best - mu) / sigma
    # Numerically stable: for very negative z, cdf ~ 0, pdf ~ 0.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Lower Confidence Bound (exploration) ---
    # LCB = mu - beta * sigma, where beta controls exploration.
    # The "improvement potential" from LCB: f_best - (mu - beta*sigma)
    # = (f_best - mu) + beta * sigma
    # Adaptive beta: larger when f_best is high (early, need to explore),
    # smaller when f_best is low (late, exploit).
    f_best_abs = abs(f_best)
    # beta ranges from ~2.0 (early) to ~0.5 (late)
    beta = 0.5 + 1.5 * np.clip(1.0 - f_best_abs / 15.0, 0.0, 1.0)
    lcb_improvement = (f_best - mu) + beta * sigma
    lcb_improvement = np.maximum(lcb_improvement, 0.0)

    # --- Normalize both terms to comparable scale ---
    # Use the max of each term as a robust scale (avoids division by zero).
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    lcb_max = np.max(lcb_improvement) if np.max(lcb_improvement) > 1e-12 else 1.0
    ei_norm = ei / ei_max
    lcb_norm = lcb_improvement / lcb_max

    # --- Cost-aware weighting ---
    # Use a softer cost penalty: cost^0.4 instead of cost^0.5
    cost_penalty = np.power(cost, 0.4)
    ei_cost_aware = ei_norm / cost_penalty
    lcb_cost_aware = lcb_norm / cost_penalty

    # --- Dynamic blend weight ---
    # Exploration weight increases as f_best decreases (closer to optimum).
    weight_explore = np.clip(1.0 - f_best_abs / 10.0, 0.0, 1.0)
    # Clamp to [0.1, 0.9] to ensure both terms always contribute.
    weight_explore = np.clip(weight_explore, 0.1, 0.9)

    # --- Combine ---
    score = (1.0 - weight_explore) * ei_cost_aware + weight_explore * lcb_cost_aware

    # --- Small cheapness bonus as tiebreaker ---
    cheap_bonus = 0.02 * cheapness
    score = score + cheap_bonus

    # Ensure finite, non-negative output.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
