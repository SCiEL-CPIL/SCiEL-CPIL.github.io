# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Uncertainty-Weighted Improvement (UWI).
    
    Instead of using a single global f_best baseline, we estimate a local baseline
    for each candidate based on the GP mean predictions of nearby candidates.
    The UWI score rewards candidates where the GP posterior variance (sigma) is high
    relative to the local prediction uncertainty, targeting boundaries of explored basins.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions for 2000 candidates
    sigma : array of shape (2000,)
        GP standard deviation predictions for 2000 candidates
    f_best : float
        Lowest observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate local baseline for each candidate using kernel density weighting
    # of the GP means. We use a simple approach: for each candidate, the local baseline
    # is a weighted average of all GP means, with weights based on how similar the
    # GP predictions are (using mu as a proxy for location in the function space).
    
    # To keep this computationally efficient with 2000 candidates, we use a
    # distance-based kernel in the (mu, sigma) space.
    
    # Compute pairwise distances in the (mu, sigma) feature space
    # But 2000x2000 might be heavy; let's use a simpler approach.
    
    # Alternative: Use the fact that we want to find candidates where the local
    # minimum of mu is close to f_best but sigma is high (boundary of explored basins).
    
    # Let's compute a local baseline as a rolling/neighborhood minimum of mu.
    # Since we don't have coordinates, we use the GP mean values themselves to
    # define neighborhoods. Candidates with similar mu values are considered nearby.
    
    # Sort candidates by mu value
    sorted_idx = np.argsort(mu)
    sorted_mu = mu[sorted_idx]
    sorted_sigma = sigma[sorted_idx]
    sorted_cost = cost[sorted_idx]
    
    # For each candidate, estimate a local baseline as the minimum mu within a
    # small window in the sorted order. This approximates the local minimum of the
    # GP mean around that candidate.
    window_size = max(10, n // 50)  # ~40 neighbors
    
    local_baseline = np.zeros(n)
    for i in range(n):
        # Find indices within window_size in the sorted order
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        local_baseline[i] = np.min(sorted_mu[start:end])
    
    # Map back to original order
    local_baseline_orig = np.zeros(n)
    local_baseline_orig[sorted_idx] = local_baseline
    
    # Compute Uncertainty-Weighted Improvement (UWI)
    # UWI = (local_baseline - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # where z = (local_baseline - mu) / sigma
    # This is similar to EI but with a local baseline instead of global f_best.
    
    # The local baseline should be closer to f_best than f_best itself for candidates
    # near the optimum, but for candidates far from the optimum, the local baseline
    # will be higher (worse), making the improvement smaller.
    
    # To ensure we still explore unexplored regions, we add a term that rewards
    # high sigma relative to the local prediction uncertainty.
    
    z = (local_baseline_orig - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Standard EI with local baseline
    ei_local = (local_baseline_orig - mu) * cdf + sigma * pdf
    
    # Uncertainty weighting: reward candidates where sigma is high relative to
    # the scale of the local improvement. This targets boundaries of explored basins.
    # The idea is that if sigma is large compared to the expected improvement,
    # there's more to learn.
    
    # A simple uncertainty weight: sigma / (sigma + abs(ei_local))
    # This is 1 when ei_local is small (high uncertainty relative to improvement)
    # and approaches 0 when ei_local is large (low uncertainty relative to improvement)
    
    uncertainty_weight = sigma / (sigma + np.abs(ei_local) + 1e-9)
    
    # Combine EI and uncertainty
    uwi = ei_local * (1 + uncertainty_weight)
    
    # Also consider the global f_best to ensure we don't miss the global optimum
    # Blend local and global baselines
    z_global = (f_best - mu) / sigma
    cdf_global = norm.cdf(z_global)
    pdf_global = norm.pdf(z_global)
    ei_global = (f_best - mu) * cdf_global + sigma * pdf_global
    
    # Blend: 70% local, 30% global
    blended_ei = 0.7 * ei_local + 0.3 * ei_global
    
    # Final score: blended EI times uncertainty weight, divided by cost
    scores = blended_ei * (1 + uncertainty_weight) / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
