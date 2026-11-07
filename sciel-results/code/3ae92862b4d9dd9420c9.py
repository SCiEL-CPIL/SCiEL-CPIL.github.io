"""Acquisition function for Bayesian optimization of Beale 2D.

Uses quantile-binned exploration: within each bin of predicted means,
reward high sigma to force systematic coverage of diverse regions,
preventing clustering in local basins on Beale's narrow valley.
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
    z = (f_best - mu) / (sigma + 1e-12)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety: ensure finite, positive sigma.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # EI component for exploitation.
    ei = _expected_improvement(mu, sigma, f_best)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Quantile-binned exploration:
    # 1. Bin the 2000 points by their predicted mean (mu) into K bins.
    # 2. Within each bin, score by sigma (uncertainty) to encourage
    #    exploring uncertain regions across the entire posterior-mean space.
    # This prevents the acquisition from collapsing to a single local basin.
    
    n_bins = 20
    # Assign each point to a bin based on its mu rank.
    mu_sorted_idx = np.argsort(mu)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    
    bin_scores = np.zeros(n)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if lo >= hi:
            continue
        idx_in_bin = mu_sorted_idx[lo:hi]
        sigma_in_bin = sigma[idx_in_bin]
        
        # Within this bin, score by sigma (higher uncertainty = higher score)
        # Normalize within bin to [0, 1]
        s_lo = float(np.min(sigma_in_bin))
        s_hi = float(np.max(sigma_in_bin))
        if (s_hi - s_lo) < 1e-12:
            bin_scores[idx_in_bin] = 1.0
        else:
            bin_scores[idx_in_bin] = (sigma_in_bin - s_lo) / (s_hi - s_lo)
    
    # Progress proxy: shift from exploration to exploitation as optimization proceeds.
    # Use the gap between f_best and min(mu) as a proxy.
    mu_min = float(np.min(mu))
    gap = f_best - mu_min
    # Normalize gap: early on gap is large, late on gap is small.
    # Scale: Beale's range is roughly [0, ~16], so gap > 8 means early.
    explore_weight = 0.3 + 0.7 * min(1.0, max(0.0, gap / 8.0))
    exploit_weight = 1.0 - explore_weight

    # Normalize EI to [0, 1]
    ei_lo = float(np.min(ei))
    ei_hi = float(np.max(ei))
    if (ei_hi - ei_lo) < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_lo) / (ei_hi - ei_lo)

    score = exploit_weight * ei_norm + explore_weight * bin_scores

    # Final numerical safety: finite, higher = evaluate next.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
