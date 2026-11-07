"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330 s wall-clock limit; keep a safety margin)
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0  # seconds before the hard limit


def _within_deadline():
    return (time.time() - _T0) < (_DEADLINE - _MARGIN)


def acquisition(mu, sigma, f_best):
    """Hybrid exploitation + exploration acquisition.

    mu:    array of predicted means, shape (N,)
    sigma: array of predicted std devs, shape (N,)
    f_best: lowest observed value so far

    Returns array of scores, shape (N,), higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # Numerical safeguards
    eps = 1e-12
    sigma = np.maximum(sigma, eps)

    # Regret scale: how much room is there to improve?
    # Use a small positive floor so early on (f_best large) we don't over-exploit.
    regret_scale = max(abs(f_best) * 1e-6, 1e-3)

    # --- Exploitation term: expected improvement (closed form) ---
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical overflow in cdf/pdf
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Exploration term: sigma scaled by a logistic gate ---
    # When uncertainty is large relative to regret scale, explore more.
    sigma_ratio = sigma / (regret_scale + sigma)
    # logistic gate: smooth transition from 0 (exploit) to 1 (explore)
    # centered so that moderate uncertainty gives moderate exploration
    explore_gate = 1.0 / (1.0 + np.exp(-3.0 * (sigma_ratio - 0.3)))

    # Upper confidence bound component (exploration)
    # kappa adapts: larger when we have fewer effective samples (early)
    # Here we use a fixed moderate kappa but scale by the gate
    kappa = 1.5
    ucb = -mu + kappa * sigma * explore_gate

    # Normalize both terms to comparable scale
    # Normalize EI by its max (or a floor)
    ei_max = np.max(ei)
    if ei_max < eps:
        ei_max = 1.0
    ei_norm = ei / ei_max

    ucb_max = np.max(np.abs(ucb))
    if ucb_max < eps:
        ucb_max = 1.0
    ucb_norm = ucb / ucb_max

    # Blend: weight shifts from exploration to exploitation as uncertainty drops
    # Use the average sigma as a proxy for global uncertainty
    avg_sigma_ratio = np.mean(sigma_ratio)
    # weight for exploitation increases as avg uncertainty decreases
    exploit_weight = 1.0 / (1.0 + np.exp(-4.0 * (0.5 - avg_sigma_ratio)))
    explore_weight = 1.0 - exploit_weight

    scores = exploit_weight * ei_norm + explore_weight * ucb_norm

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
