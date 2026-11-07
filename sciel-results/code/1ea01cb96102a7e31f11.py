"""Acquisition function for Bayesian optimization of Beale 2D.

Quantile Regression Forest-style aggregation: partition the 2000 candidates
into disjoint quantile bins of the posterior mean. Within each bin, every
point receives the maximum sigma (uncertainty) observed in that bin. This
forces a systematic, grid-based exploration of high-uncertainty regions
across the entire posterior-mean space, preventing the clustering of
evaluations seen in standard EI/LCB methods on Beale's narrow valley.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_START = time.time()
_SAFETY_MARGIN = 25.0


def _time_left():
    return 300.0 - (time.time() - _DEADLINE_START) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety: ensure finite, positive sigma.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Quantile Regression Forest-style aggregation:
    # 1. Sort candidates by predicted mean (mu).
    # 2. Partition into K disjoint quantile bins.
    # 3. Within each bin, assign every point the maximum sigma in that bin.
    # This forces systematic coverage: the highest-uncertainty region in each
    # quantile of the posterior-mean space is explored, preventing clustering.
    
    n_bins = 20
    mu_sorted_idx = np.argsort(mu)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    
    score = np.zeros(n)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if lo >= hi:
            continue
        idx_in_bin = mu_sorted_idx[lo:hi]
        # Maximum sigma within this bin becomes the score for all points in the bin.
        max_sigma_in_bin = float(np.max(sigma[idx_in_bin]))
        score[idx_in_bin] = max_sigma_in_bin

    # Final numerical safety: finite, higher = evaluate next.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
