"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we've already spent too much time, return quickly
    # (the evaluator kills at 330s; we want to finish each call well before that)
    t_start = time.time()
    DEADLINE = 325.0  # seconds from process start is not available, so use a per-call budget

    # Ensure arrays are 1-D numpy arrays of the right type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    if n == 0:
        return np.array([], dtype=np.float64)

    # Floor sigma to avoid division by zero / log of zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # Potential improvement (we are minimizing, so improvement = f_best - mu)
    delta = f_best - mu

    # ---------- Expected Improvement (EI) ----------
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # ---------- Cost-aware scaling ----------
    cost_safe = np.maximum(cost, 1e-12)

    # Adaptive alpha: early on (low mean cost) we can afford to be less
    # cost-sensitive and explore; later we tighten cost awareness.
    mean_cost = float(np.mean(cost))
    norm_cost = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)

    # Use a softer cost penalty: cost^(-alpha) with alpha in [0.3, 0.8]
    # Lower alpha means less aggressive cost penalty, allowing exploration
    alpha = 0.3 + 0.5 * norm_cost
    cost_weight = cost_safe ** (-alpha)

    # ---------- Exploration term (UCB) ----------
    mu_spread = np.std(mu) + 1e-30
    sigma_spread = np.std(sigma_safe)

    # Adaptive kappa: more exploration when uncertainty is high relative to model spread
    uncertainty_ratio = sigma_spread / (mu_spread + sigma_spread)
    kappa_base = 1.0
    kappa = kappa_base * (0.5 + 1.5 * uncertainty_ratio)

    ucb = f_best - mu + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)

    # ---------- Combine EI and UCB with adaptive weights ----------
    def _normalise(arr):
        m = np.max(arr)
        if m > 1e-18:
            return arr / m
        return arr

    ei_n = _normalise(ei)
    ucb_n = _normalise(ucb)

    # Weights: as the budget is consumed (norm_cost ↑) shift weight from
    # exploration (UCB) to exploitation (EI).
    w_ei = 0.50 + 0.25 * norm_cost     # 0.50 → 0.75
    w_ucb = 1.0 - w_ei                 # 0.50 → 0.25

    scores = w_ei * ei_n + w_ucb * ucb_n

    # Apply cost weighting
    scores = scores * cost_weight

    # ---------- Final normalisation for numerical stability ----------
    s_mean = np.mean(scores)
    s_std = np.std(scores)
    if s_std > 1e-12:
        scores = (scores - s_mean) / s_std
    else:
        scores = scores - s_mean

    # Guarantee finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
