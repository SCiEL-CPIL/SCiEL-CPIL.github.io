"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START_TIME = time.time()
_ACQ_DEADLINE = _ACQ_START_TIME + 280


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
    # Deadline guard
    if time.time() > _ACQ_DEADLINE:
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Sanitize
    sigma = np.maximum(np_sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Reference scale for numerical stability
    ref = max(abs(f_best), 1e-6)
    z = (f_best - mu) / sigma

    # --- Smooth EI with soft-min surrogate ---
    # Instead of hard EI = max(0, (f_best - mu) * Phi(z) + sigma * phi(z)),
    # use a soft-min to smooth the kink at z=0 (where improvement crosses 0).
    # The standard EI is already smooth for z > 0 and 0 for z < 0, but the
    # transition can be sharpened by the CDF/PDF. We compute EI directly and
    # apply a soft-min with a small temperature to ensure smoothness.

    z_clipped = np.clip(z, -20, 20)
    cdf_vals = norm.cdf(z_clipped)
    pdf_vals = norm.pdf(z_clipped)

    # Raw EI (can be slightly negative due to numerical issues, clip to 0)
    ei_raw = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei_raw = np.maximum(ei_raw, 0.0)

    # Soft-min surrogate: smooth the maximum(ei, 0) via log-sum-exp
    # soft_min(a, b, t) = -logsumexp(-a/t, -b/t) * t
    # Here we want soft_max(0, ei_raw) which is just a smooth ReLU-like
    # but since ei_raw >= 0 already, we use a gentle smoothing on the
    # improvement term to avoid any residual kink.
    # Temperature for soft-min: small relative to ref
    tau = 0.05 * ref
    # soft_relu(x) = log(1 + exp(x/tau)) * tau  (smooth max(0, x))
    # But ei_raw is already >= 0, so we just ensure smoothness in z-space.
    # Actually, the EI formula is already smooth; the "kink" is in the
    # indicator. We keep ei_raw as-is but add a tiny epsilon for stability.
    ei = ei_raw + 1e-12

    # --- Smooth cost-aware penalty via log-sum-exp epigraph ---
    # Instead of hard division ei / cost, use a smooth penalty:
    # score = ei - lambda * logsumexp(cost/tau_c, 0) * something
    # Better: use a smooth "cost efficiency" that penalizes high cost
    # without the cliff of division.
    #
    # Epigraph approach: minimize ei - s.t. cost <= c
    # Lagrangian dual gives: ei - max(0, lambda * (cost - c))
    # Smooth version: ei - lambda * softplus(cost - c)
    #
    # We want higher score for lower cost. Use:
    # penalty = lambda * log(1 + exp((cost - c_ref) / tau_c)) * tau_c
    # This is a smooth convex penalty that is ~0 for cost << c_ref
    # and ~lambda*(cost - c_ref) for cost >> c_ref.

    c_ref = np.median(cost)  # reference cost
    tau_c = 0.1 * c_ref + 1e-9  # temperature for softplus
    lambda_cost = 0.5  # penalty strength

    # Smooth cost penalty: softplus(cost - c_ref) * lambda
    # softplus(x) = log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
    x_cost = (cost - c_ref) / tau_c
    # Numerically stable softplus
    softplus = np.where(x_cost > 20, x_cost * tau_c,
                        tau_c * np.log1p(np.exp(x_cost)))
    cost_penalty = lambda_cost * softplus

    # Cost-aware score: EI minus smooth cost penalty
    # Also add a small exploration term based on sigma
    sigma_norm = sigma / (sigma.max() + 1e-12)
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_norm

    score = ei - cost_penalty + exploration

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
