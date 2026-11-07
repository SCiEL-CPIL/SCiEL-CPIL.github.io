"""
Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D).

Structural change: Replace the multiplicative cost-discounted EI + UCB mixture
with a two-stage "confidence-gated" acquisition:
  - Exploit branch: cost-normalized EI (rewards high-certainty improvements per cost)
  - Explore branch: UCB-like sigma/cost (rewards high-uncertainty, low-cost points)
  - Gate: a soft sigmoid that shifts weight toward exploitation as the GP becomes
    more confident (low sigma relative to problem scale), without hard thresholds.

This is a different modelling family: instead of a single linear combination with
adaptive weights tuned to EI concentration, we use a probabilistic gating mechanism
that mimics a posterior over "the optimum is near" vs. "we need more data".
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: keep a safety margin before the 330 s wall-clock kill
_DEADLINE = time.time() + 315.0


def _safe_norm_pdf(z):
    """Numerically stable norm.pdf."""
    return norm.pdf(z)


def _safe_norm_cdf(z):
    """Numerically stable norm.cdf."""
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.

    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate

    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Guard against zero/negative values
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # --- Expected Improvement (standard) ---
    delta = f_best - mu
    z = delta / sigma
    phi_z = _safe_norm_pdf(z)
    Phi_z = _safe_norm_cdf(z)
    ei = delta * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)

    # --- Cost-normalized EI (exploit) ---
    ei_cost = ei / cost

    # --- UCB-like exploration term ---
    ucb_term = sigma / cost

    # --- Confidence gate: probability that the GP is "confident enough" to exploit ---
    # Use the ratio of mean sigma to the problem scale (|f_best| + 1) as a confidence proxy.
    # When sigma is small relative to the problem, we are confident -> exploit.
    # When sigma is large, we are uncertain -> explore.
    problem_scale = abs(f_best) + 1.0
    mean_sigma = np.mean(sigma)
    # Gate in [0, 1]: 1 = fully exploit, 0 = fully explore
    # Sigmoid centered at 0.1 * problem_scale (typical sigma when converged)
    gate_arg = (0.1 * problem_scale - mean_sigma) / (0.05 * problem_scale + 1e-9)
    gate = 1.0 / (1.0 + np.exp(-gate_arg))  # sigmoid
    # Clip to [0.1, 0.9] to always keep some of both
    gate = np.clip(gate, 0.1, 0.9)

    # --- Combine via gated mixture ---
    # Exploit weight = gate, explore weight = 1 - gate
    w_exploit = gate
    w_explore = 1.0 - gate

    score = w_exploit * ei_cost + w_explore * ucb_term

    # --- Secondary boost: reward candidates where EI is high AND cost is low ---
    # This is a multiplicative synergy term that is zero when either EI or low-cost is absent.
    # Normalize EI and inverse-cost to [0,1] for a bounded boost.
    ei_max = np.max(ei_cost)
    if ei_max > 1e-12:
        ei_norm = ei_cost / ei_max
    else:
        ei_norm = np.zeros(n)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min + 1e-9:
        inv_cost_norm = 1.0 - (cost - cost_min) / (cost_max - cost_min)
    else:
        inv_cost_norm = np.ones(n)
    synergy = ei_norm * inv_cost_norm
    score += 0.15 * synergy

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
