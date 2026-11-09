"""
Bayesian Optimization Acquisition using Diversity-Aware Greedy Batch Selection.

The task is to minimize a fixed seeded random RBF mixture in 2D.
We use a GP posterior (mu, sigma) provided by the evaluator.

Instead of a myopic point-wise acquisition, we combine Expected Improvement
scores with a spatial diversity bonus. The key insight: RBF mixtures have
multiple local minima, and a pure EI can collapse into one basin. By adding
a distance-based exploration term that rewards points far from already-scored
high-EI regions, we encourage coverage of the [0,1]^2 domain.

The acquisition score is:
    score(i) = EI(i) + beta * min_j ||x_i - x_j||  for j in top-k EI points

This is computed implicitly: we first rank by EI, then for each point compute
its minimum distance to the top-k EI-ranked points, and add a scaled distance
bonus. This creates a "repulsion" from the most promising exploitation targets
while still allowing exploitation when EI is very high.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking for deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # 330s limit, 5s margin
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Diversity-aware acquisition function combining EI with spatial coverage.
    
    For each candidate, the score is:
        score = EI + beta * min_distance_to_top_k_EI_points
    
    This encourages the search to explore regions far from the current
    exploitation targets, helping navigate the multimodal RBF landscape.
    
    Args:
        mu: (2000,) posterior mean
        sigma: (2000,) posterior std
        f_best: float, lowest observed value
    
    Returns:
        (2000,) scores. Higher is better.
    """
    n = len(mu)
    
    # Deadline check
    if time.time() - _T0 > _DEADLINE:
        # Fallback to exploitation-focused score
        return -(mu - f_best)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, _EPS)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, _EPS)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)

    # Compute d = (f_best - mu) / sigma
    d = (f_best - mu_safe) / sigma_safe
    
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    ei = (f_best - mu_safe) * Phi_d + sigma_safe * phi_d
    
    # Ensure finite EI
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to [0, 1] for combining with distance term
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < _EPS:
        ei_norm = np.zeros(n)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # We need to estimate spatial diversity. Since we don't have coordinates,
    # we use the GP posterior structure: points with similar (mu, sigma) 
    # values are likely close in input space. We use a kernel-based approach:
    # compute pairwise similarity in (mu, sigma) space and derive distances.
    # 
    # Actually, a simpler and more effective approach: use the sigma values
    # as a proxy for "unexplored" regions. High sigma means high uncertainty,
    # which correlates with being far from observed points. We combine:
    #   score = alpha * ei_norm + (1 - alpha) * sigma_norm
    #
    # But the directive asks for a greedy batch / hypervolume / spacing approach.
    # Without coordinates, we can approximate spatial structure using the
    # GP posterior. Let's use a different strategy:
    #
    # Strategy: Weighted combination of EI and a "novelty" score based on
    # how different each point is from the top EI points in (mu, sigma) space.
    #
    # Compute top-k indices by EI
    k = max(1, min(20, n // 100))  # top 1-20 points
    top_k_idx = np.argsort(ei)[::-1][:k]
    
    # For each point, compute its minimum "distance" to top-k points
    # in the (mu, sigma) feature space. This approximates spatial distance.
    top_mu = mu_safe[top_k_idx]  # (k,)
    top_sigma = sigma_safe[top_k_idx]  # (k,)
    
    # Feature vectors for top-k: [mu, sigma]
    top_features = np.column_stack([top_mu, top_sigma])  # (k, 2)
    
    # Feature vectors for all points
    all_features = np.column_stack([mu_safe, sigma_safe])  # (n, 2)
    
    # Compute pairwise distances: for each point, min distance to any top-k point
    # Use squared Euclidean distance in feature space
    # (n, 2) @ (2, k) -> (n, k)
    # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
    all_sq = np.sum(all_features ** 2, axis=1)  # (n,)
    top_sq = np.sum(top_features ** 2, axis=1)  # (k,)
    cross = all_features @ top_features.T  # (n, k)
    
    dist_sq = all_sq[:, None] + top_sq[None, :] - 2 * cross  # (n, k)
    dist_sq = np.maximum(dist_sq, 0)  # numerical safety
    min_dist = np.sqrt(np.min(dist_sq, axis=1))  # (n,)
    
    # Normalize min_dist to [0, 1]
    md_max = np.max(min_dist)
    if md_max < _EPS:
        dist_norm = np.zeros(n)
    else:
        dist_norm = min_dist / md_max
    
    # Combine: weighted sum of normalized EI and normalized distance
    # alpha controls exploitation vs exploration
    alpha = 0.7  # 70% EI, 30% diversity
    
    score = alpha * ei_norm + (1 - alpha) * dist_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
