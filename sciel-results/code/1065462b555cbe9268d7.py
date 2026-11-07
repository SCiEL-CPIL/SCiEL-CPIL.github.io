"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 310  # 310s out of 330s budget, safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Uncertainty-Weighted Improvement (UWI).
    
    Structural change: Instead of a single global f_best baseline, we estimate
    a local baseline via kernel density weighting of the GP predictions, and
    compute a UWI score that rewards candidates where the GP posterior variance
    is high relative to the local prediction, targeting basin boundaries.
    
    The UWI score combines:
    1. Expected improvement relative to a local (not global) baseline
    2. An uncertainty bonus that scales with sigma relative to local scale
    3. Cost normalization to prefer cheaper evaluations
    """
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate a local baseline using kernel density weighting of mu values.
    # We use the GP mean predictions themselves to create a soft local minimum
    # that adapts to the landscape structure.
    # 
    # For each candidate, the local baseline is a weighted average of nearby
    # mu values, where weights decay with distance in the mu-space.
    # This is approximated by a running minimum with softening.
    
    # Compute a local baseline: for each point, use a soft minimum over
    # the entire set of mu values, weighted by proximity in mu-space.
    # This is O(n^2) which could be slow for n=2000, so we use a binned approach.
    
    # Binned approach: divide mu into bins and compute local minima
    n_bins = 20
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    
    # Bin edges
    bin_edges = np.linspace(mu_min - 0.1 * mu_range, mu_max + 0.1 * mu_range, n_bins + 1)
    bin_indices = np.digitize(mu, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # For each bin, compute the minimum and mean of mu values in that bin
    bin_min = np.full(n_bins, np.inf)
    bin_mean = np.full(n_bins, 0.0)
    bin_count = np.zeros(n_bins)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.any(mask):
            bin_min[b] = np.min(mu[mask])
            bin_mean[b] = np.mean(mu[mask])
            bin_count[b] = np.sum(mask)
    
    # For each candidate, compute a local baseline as a weighted combination
    # of the minimum in its bin and neighboring bins
    local_baseline = np.zeros(n)
    for i in range(n):
        b = bin_indices[i]
        # Weight current bin heavily, neighbors less
        weights = [1.0, 0.3, 0.1]
        vals = []
        for offset, w in zip([0, -1, 1], weights):
            nb = b + offset
            if 0 <= nb < n_bins and bin_count[nb] > 0:
                vals.append(w * bin_min[nb])
        if vals:
            total_w = sum(weights[j] for j in range(len(vals)))
            local_baseline[i] = np.sum(vals) / total_w
        else:
            local_baseline[i] = f_best
    
    # Ensure local baseline doesn't exceed f_best (we can't do better than global best)
    local_baseline = np.minimum(local_baseline, f_best)
    
    # Compute z-score for expected improvement relative to local baseline
    z = (local_baseline - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement relative to local baseline
    ei_local = (local_baseline - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_local = np.maximum(ei_local, 0.0)
    
    # Uncertainty-Weighted Improvement: reward high variance relative to local scale
    # The key idea: where sigma is large relative to the local spread of predictions,
    # we are likely at the boundary of an explored basin
    local_scale = np.maximum(np.abs(local_baseline - mu), 1e-6)
    uncertainty_ratio = sigma_safe / (local_scale + sigma_safe)
    
    # UWI score: combines local EI with uncertainty bonus
    # Higher uncertainty_ratio means more exploration value
    uwi = ei_local + 2.0 * sigma_safe * uncertainty_ratio
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = uwi / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _start_time:
        return score
    
    return score
# EVOLVE-BLOCK-END
