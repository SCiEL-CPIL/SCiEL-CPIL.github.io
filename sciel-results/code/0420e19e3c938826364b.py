# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.

    Uses adaptive LCB: score = -(mu + beta * sigma) / cost
    where beta adapts to the current uncertainty level.

    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Hartmann 3d range: min ≈ -3.86, max ≈ 3.0 (roughly [-4, 3])
    # Target range for scaling
    min_val = -3.86278
    max_val = 3.0
    target_range = max_val - min_val  # ≈ 6.86

    # Adaptive beta: scale exploration with relative uncertainty
    # When the GP is uncertain (sigma large relative to target range),
    # we want high beta (explore). When certain, low beta (exploit).
    #
    # Use the median sigma as a robust measure of current uncertainty.
    # sigma_median / target_range gives relative uncertainty in [0, ~1]
    sigma_median = np.median(sigma_safe)
    rel_uncertainty = sigma_median / target_range  # typically 0.01 to 1.0

    # Beta: starts high (~2.5) when uncertain, drops to ~0.5 when certain
    # Smooth sigmoid-like transition
    # rel_uncertainty ~ 0.05 -> beta ~ 0.7
    # rel_uncertainty ~ 0.5  -> beta ~ 1.5
    # rel_uncertainty ~ 1.0  -> beta ~ 2.5
    beta_min = 0.5
    beta_max = 2.5
    # Linear interpolation with clamping
    beta = beta_min + (beta_max - beta_min) * np.clip(rel_uncertainty, 0.0, 1.0)

    # LCB for minimization: lower is better
    # lcb = mu + beta * sigma  (we want to minimize this)
    # Score = -lcb / cost (higher score = more desirable)
    lcb = mu_safe + beta * sigma_safe
    scores = -lcb / cost_safe

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    if time.time() > deadline:
        return scores

    # Optional: blend in a small EI component for exploitation precision
    # when uncertainty is low (late stage). This helps fine-tune around
    # promising regions without being overly conservative.
    if rel_uncertainty < 0.15 and time.time() < deadline - 5.0:
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        ei_scores = ei / cost_safe
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)

        # Normalize both to [0, 1] range for blending
        s_min, s_max = np.min(scores), np.max(scores)
        e_min, e_max = np.min(ei_scores), np.max(ei_scores)
        if s_max - s_min > 1e-12:
            scores_norm = (scores - s_min) / (s_max - s_min)
        else:
            scores_norm = np.zeros_like(scores)
        if e_max - e_min > 1e-12:
            ei_norm = (ei_scores - e_min) / (e_max - e_min)
        else:
            ei_norm = np.zeros_like(ei_scores)

        # Blend: 70% LCB + 30% EI when very certain
        blend_w = 0.3
        scores = (1.0 - blend_w) * scores_norm + blend_w * ei_norm

    # Final safety
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
