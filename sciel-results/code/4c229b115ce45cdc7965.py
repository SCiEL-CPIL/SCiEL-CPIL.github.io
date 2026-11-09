"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_SESSION_START = time.time()


def _expected_improvement_cost_aware(mu, sigma, f_best, cost):
    """
    Core acquisition: expected improvement per unit cost with
    (a) an adaptive improvement threshold (quantile-aware EI),
    (b) a super-linear cost-risk discount, and
    (c) an uncertainty-gated exploration (LCB) term.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # --- (a) Adaptive improvement threshold -------------------------------
    # Use a quantile-adjusted threshold: theta = f_best - c * sigma_median.
    # When the GP is uncertain (large sigma_median), theta drops below f_best,
    # so we only count improvements beyond a meaningful gap. When the GP is
    # confident, theta ≈ f_best and we behave like standard EI.
    sigma_median = float(np.median(sigma_safe))
    c_thresh = 0.3
    theta = f_best - c_thresh * sigma_median
    # Clamp theta to not go below 0 (rastrigin minimum) to avoid absurd thresholds
    theta = max(theta, 0.0)

    # Standard EI for minimization with threshold theta
    # EI = (theta - mu) * Phi(z) + sigma * phi(z),  z = (theta - mu) / sigma
    z = (theta - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (theta - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)  # EI is non-negative by definition

    # --- (b) Super-linear cost-risk discount -------------------------------
    # cost is in [1, 2]. We penalize expensive candidates more than linearly.
    # risk_factor = 1 + beta * (cost - 1), so cost=1 -> 1, cost=2 -> 1+beta
    # With beta=1.0, a cost-2 point is discounted by 2x more than cost-1.
    beta_cost = 1.0
    risk_factor = 1.0 + beta_cost * (cost_safe - 1.0)
    ei_cost = ei / risk_factor

    # --- (c) Uncertainty-gated exploration (LCB) ---------------------------
    # LCB score = -mu + kappa * sigma (higher = more promising / uncertain)
    # Weight by exp(-f_best / scale): early (f_best large) weight ~ small,
    # but we want MORE exploration early. So weight = exp(-f_best/scale)
    # is ~0 when f_best is large (early) — that's backwards.
    # Instead: weight = exp(f_best / scale) is huge early. Clip it.
    # Better: use a smooth step that is ~1 early, ~0 late.
    # f_best_norm = clip(f_best / 30, 0, 1); weight = f_best_norm
    # This is simple: early f_best ~ 20-40 -> weight ~ 0.67-1.0
    # late f_best ~ 0-5 -> weight ~ 0-0.17
    f_best_norm = float(np.clip(f_best / 30.0, 0.0, 1.0))
    kappa = 2.0
    lcb_score = -mu + kappa * sigma_safe
    lcb_cost = lcb_score / risk_factor

    # Blend: exploration weight = f_best_norm, exploitation weight = 1 - f_best_norm
    # This gives pure exploitation late, pure exploration early.
    # To avoid pure extremes, floor/ceiling the weights:
    w_explore = 0.3 + 0.7 * f_best_norm   # ranges [0.3, 1.0]
    w_exploit = 1.0 - w_explore           # ranges [0.0, 0.7]

    score = w_exploit * ei_cost + w_explore * lcb_cost

    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.

    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard
    elapsed = time.time() - _SESSION_START
    if elapsed > 290:
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        score = ei / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    score = _expected_improvement_cost_aware(mu, sigma, f_best, cost)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
