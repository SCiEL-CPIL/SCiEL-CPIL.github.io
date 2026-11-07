"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard: the evaluator kills the program after 330 s.
# We keep a safety margin and cap the work inside each acquisition call.
_DEADLINE = time.monotonic() + 325.0   # 5 s margin before the 330 s wall-clock limit
_CALL_BUDGET = 1.5                     # max seconds for a single acquisition call


def _norm_cdf(x):
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (cEI) blended with an adaptive UCB term.
    mu, sigma, f_best, cost: 1-D arrays of shape (2000,).
    Returns 2000 finite scores; higher = evaluate next.
    """
    t0 = time.monotonic()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    n = mu.size
    if n == 0:
        return np.empty(0, dtype=float)

    # Guard against zero/negative sigma
    sig = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Standard normal CDF/PDF for EI (use numerically stable erf-based forms)
    z = (f_best - mu) / sig
    z = np.clip(z, -10.0, 10.0)
    cdf = _norm_cdf(z)
    pdf = _norm_pdf(z)

    # Expected Improvement: E[max(f_best - f(x), 0)]  (minimization)
    ei = (f_best - mu) * cdf + sig * pdf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: divide by cost (higher cost = lower priority)
    c_ei = ei / cost_safe

    # --- Adaptive exploration (UCB blend) ---
    # Remaining budget ratio: as total cost approaches 30, we explore less.
    # cost here is the per-candidate evaluation cost (in [1, 2]); use its mean
    # as a proxy for the current spending rate. The loop stops at total cost 30.
    # We don't know total cost directly, but a robust proxy: the mean of the
    # per-candidate costs is ~1.5 (uniform in [0,1] scaled). Instead, use the
    # magnitude of EI relative to sigma to set exploration, which naturally
    # adapts: early on sigma is large everywhere (little data) -> more UCB.
    # We scale UCB by a beta that shrinks as EI becomes informative.
    # beta ~ 1 / (1 + mean(EI)/ (mean(sigma)+eps))  -> large when EI small vs sigma
    mean_ei = float(np.mean(ei)) if n else 0.0
    mean_sig = float(np.mean(sig)) if n else 1.0
    ratio = mean_ei / (mean_sig + 1e-9)
    beta = 1.0 / (1.0 + 1.5 * ratio)     # in (0, 1]; high when exploration needed

    # UCB term: mu is low (good) and sigma is high (uncertain) -> high score.
    # Normalize mu and sigma to [0,1] range for a stable blend.
    mu_lo, mu_hi = float(np.min(mu)), float(np.max(mu))
    sig_lo, sig_hi = float(np.min(sig)), float(np.max(sig))
    mu_norm = (mu - mu_lo) / (mu_hi - mu_lo + 1e-9)
    sig_norm = (sig - sig_lo) / (sig_hi - sig_lo + 1e-9)
    ucb = (1.0 - mu_norm) + sig_norm     # prefer low predicted value + high uncertainty

    # Normalize c_ei to [0,1] for a comparable blend
    ce_lo, ce_hi = float(np.min(c_ei)), float(np.max(c_ei))
    c_ei_norm = (c_ei - ce_lo) / (ce_hi - ce_lo + 1e-9)

    # Blend: exploitation (cost-aware EI) + exploration (UCB)
    scores = (1.0 - beta) * c_ei_norm + beta * ucb

    # Small penalty for already-explored (low sigma) points to encourage diversity
    # but keep it gentle so it doesn't override the main signal.
    scores = scores * (0.9 + 0.1 * sig_norm)

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard: if we are running long, return immediately (already computed).
    if (time.monotonic() - t0) > _CALL_BUDGET:
        return scores

    return scores
# EVOLVE-BLOCK-END
