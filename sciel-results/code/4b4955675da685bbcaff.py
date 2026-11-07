"""Acquisition function for Bayesian optimization of Beale 2D.

Combines quantile-binned exploration (structural idea from current program)
with an EI-dominant blending scheme and adaptive progress proxy (refinement
from inspirations). The key insight: bin by posterior mean to force
systematic coverage, but within each bin use a hybrid of normalized EI
(exploitation) and normalized sigma (exploration) rather than pure sigma.
This preserves the anti-clustering property while still allowing the
acquisition to prefer slightly better predicted means within each bin.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_START = time.time()
_SAFETY_MARGIN = 25.0


def _time_left():
    return 300.0 - (time.time() - _DEADLINE_START) - _SAFETY_MARGIN


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where sigma is ~0, EI should be max(0, f_best - mu)
    ei = np.where(sigma < eps, np.maximum(f_best - mu, 0.0), ei)
    return np.where(np.isfinite(ei), ei, 0.0)


def _normalize(a):
    """Normalize array to [0, 1] range. Returns zeros if degenerate."""
    a = np.asarray(a, dtype=float)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety: ensure finite, positive sigma.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Compute EI for all points (global exploitation signal).
    ei = _expected_improvement(mu, sigma, f_best)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Quantile-binned exploration:
    # 1. Bin the 2000 points by their predicted mean (mu) into K bins.
    # 2. Within each bin, compute a hybrid score: blend of normalized EI
    #    and normalized sigma. This forces coverage across the entire
    #    posterior-mean space while still preferring slightly better
    #    predicted means within each bin.
    
    n_bins = 25
    mu_sorted_idx = np.argsort(mu)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    
    bin_scores = np.zeros(n)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if lo >= hi:
            continue
        idx_in_bin = mu_sorted_idx[lo:hi]
        ei_in_bin = ei[idx_in_bin]
        sigma_in_bin = sigma[idx_in_bin]
        
        # Within this bin, blend EI and sigma.
        # Use a fixed intra-bin blend: 0.6 EI + 0.4 sigma
        # This is different from pure sigma (current) and from pure EI (seed).
        ei_norm_bin = _normalize(ei_in_bin)
        sigma_norm_bin = _normalize(sigma_in_bin)
        
        intra_bin_score = 0.6 * ei_norm_bin + 0.4 * sigma_norm_bin
        bin_scores[idx_in_bin] = intra_bin_score
    
    # Progress proxy: shift from exploration to exploitation as optimization proceeds.
    # Use the gap between f_best and min(mu) as a proxy.
    mu_min = float(np.min(mu))
    gap = f_best - mu_min
    # Beale's range is roughly [0, ~16]. gap > 8 means early.
    explore_weight = 0.3 + 0.7 * min(1.0, max(0.0, gap / 8.0))
    exploit_weight = 1.0 - explore_weight

    # Normalize global EI to [0, 1] for the exploitation component.
    ei_norm_global = _normalize(ei)

    # Final score: blend of global EI (exploitation) and bin scores (exploration).
    score = exploit_weight * ei_norm_global + explore_weight * bin_scores

    # Final numerical safety: finite, higher = evaluate next.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
