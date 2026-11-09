"""Cost-aware acquisition using set-based greedy selection with joint EI and cost penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s wall-clock safety margin (limit is 330s)


def _joint_ei_approx(mu_subset, sigma_subset, f_best, L):
    """
    Approximate joint expected improvement for a subset of candidates.
    Uses the fact that for a set of correlated Gaussian predictions,
    the joint EI can be approximated by summing individual EIs minus
    a redundancy penalty based on pairwise correlations.
    
    For simplicity and speed, we use:
    Joint_EI ~ sum(EI_i) - redundancy_penalty
    where redundancy_penalty captures how much information is shared.
    
    A practical approximation: if two points have similar predictions
    (high correlation), evaluating both is less valuable than evaluating
    one. We penalize based on similarity of (mu, sigma) pairs.
    """
    n = len(mu_subset)
    if n == 0:
        return 0.0
    if n == 1:
        sigma_safe = max(sigma_subset[0], 1e-9)
        z = (f_best - mu_subset[0]) / sigma_safe
        ei = (f_best - mu_subset[0]) * norm.cdf(z) + sigma_subset[0] * norm.pdf(z)
        return max(ei, 0.0)
    
    # Compute individual EIs
    sigma_safe = np.maximum(sigma_subset, 1e-9)
    z = (f_best - mu_subset) / sigma_safe
    ei = (f_best - mu_subset) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Redundancy penalty: based on similarity of predictions
    # High mu similarity + high sigma similarity = high redundancy
    # Use a simple distance-based penalty
    mu_range = np.ptp(mu_subset) if n > 1 else 0
    sigma_range = np.ptp(sigma_subset) if n > 1 else 0
    
    # Scale ranges to [0,1] for comparison
    mu_scale = max(mu_range, 1e-9)
    sigma_scale = max(sigma_range, 1e-9)
    
    # Compute pairwise similarity matrix (simplified)
    # Use mean pairwise distance as a diversity measure
    mu_diffs = np.abs(mu_subset[:, None] - mu_subset[None, :])
    sigma_diffs = np.abs(sigma_subset[:, None] - sigma_subset[None, :])
    
    # Normalize
    mu_diffs_norm = mu_diffs / mu_scale
    sigma_diffs_norm = sigma_diffs / sigma_scale
    
    # Average pairwise distance (excluding diagonal)
    mask = ~np.eye(n, dtype=bool)
    avg_mu_dist = mu_diffs_norm[mask].mean() if n > 1 else 0
    avg_sigma_dist = sigma_diffs_norm[mask].mean() if n > 1 else 0
    
    # Diversity score: higher = more diverse
    diversity = 0.5 * avg_mu_dist + 0.5 * avg_sigma_dist
    
    # Redundancy penalty: lower diversity = higher penalty
    # Penalty scales with the number of points
    redundancy_penalty = (1 - diversity) * (n - 1) * 0.1
    
    return np.sum(ei) - redundancy_penalty


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute individual EI scores
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / cost_safe
    
    # Compute UCB scores
    beta = 2.0
    ucb = (f_best - mu) + beta * sigma_safe
    ucb_cost = ucb / cost_safe
    
    # Combine EI and UCB with adaptive weighting
    # Higher weight to UCB when uncertainty is high relative to improvement
    improvement = f_best - mu
    ratio = sigma_safe / (np.abs(improvement) + 1e-9)
    w = 0.5 * np.tanh(ratio)
    
    base_score = (1 - w) * ei_cost + w * np.maximum(ucb_cost, 0.0)
    
    # Now apply set-based greedy selection to adjust scores
    # We want to boost scores for points that add diversity to the top candidates
    # and penalize points that are redundant with other high-scoring points
    
    # Get indices sorted by base score
    sorted_idx = np.argsort(-base_score)
    
    # Greedy selection: build a diverse set of top candidates
    # For each candidate, compute a diversity-adjusted score
    
    # Use a simpler approach: compute a "redundancy penalty" based on
    # proximity to other high-scoring candidates in (mu, sigma) space
    
    # Normalize mu and sigma for distance computation
    mu_range = np.ptp(mu)
    sigma_range = np.ptp(sigma)
    mu_norm = (mu - np.min(mu)) / (mu_range + 1e-9)
    sigma_norm = (sigma - np.min(sigma)) / (sigma_range + 1e-9)
    
    # For efficiency, only consider top K candidates for diversity computation
    K = min(200, n)
    top_idx = sorted_idx[:K]
    top_mu = mu_norm[top_idx]
    top_sigma = sigma_norm[top_idx]
    top_scores = base_score[top_idx]
    
    # Compute pairwise distances in normalized (mu, sigma) space
    # dist_ij = sqrt((mu_i - mu_j)^2 + (sigma_i - sigma_j)^2)
    mu_diff = top_mu[:, None] - top_mu[None, :]
    sigma_diff = top_sigma[:, None] - top_sigma[None, :]
    dist = np.sqrt(mu_diff**2 + sigma_diff**2)
    
    # For each point, compute a diversity bonus based on minimum distance
    # to other high-scoring points
    # Points that are far from other high-scoring points get a bonus
    
    # Initialize diversity bonus
    diversity_bonus = np.zeros(n)
    
    # For top K points, compute bonus
    for i in range(K):
        # Minimum distance to other top-K points (excluding self)
        if K > 1:
            min_dist = np.min(dist[i, np.concatenate(([0]*i, [1]*(K-i-1)))])
        else:
            min_dist = 1.0
        
        # Bonus scales with minimum distance and the score of this point
        # Higher score + more diverse = higher bonus
        diversity_bonus[top_idx[i]] = min_dist * top_scores[i]
    
    # Normalize diversity bonus
    if np.max(diversity_bonus) > 1e-9:
        diversity_bonus = diversity_bonus / np.max(diversity_bonus)
    
    # Final score: base score + diversity bonus
    # The diversity bonus encourages exploring different regions
    final_score = base_score + 0.3 * diversity_bonus
    
    # Ensure all scores are finite and non-negative
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    final_score = np.maximum(final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
