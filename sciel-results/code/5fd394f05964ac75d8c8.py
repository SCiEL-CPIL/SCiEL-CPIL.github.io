"""Cost-aware acquisition for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill the program before 330s wall-clock.
_DEADLINE_SECONDS = 320.0  # safety margin
_start_time = time.time()


def _time_remaining():
    return _DEADLINE_SECONDS - (time.time() - _start_time)


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement (EI) for minimization."""
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, f_best, beta=1.0):
    """Upper Confidence Bound (UCB) for minimization: lower = more promising."""
    # For minimization, we want to explore where mu is low and sigma is high.
    # UCB for minimization: mu - beta * sigma
    return mu - beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of Improvement (PI) for minimization."""
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    pi = norm.cdf(z)
    return np.maximum(pi, 0.0)


def _kmeans_select(scores, n_clusters, max_iter=10):
    """
    Simple k-means clustering on 1D scores to select diverse candidates.
    Returns indices of cluster centers (best representative per cluster).
    """
    if n_clusters >= len(scores):
        return np.argsort(scores)[:n_clusters]
    
    # Initialize centers: pick top n_clusters by score, spread out
    # Use a greedy initialization
    idx_sorted = np.argsort(scores)[::-1]
    centers_idx = [idx_sorted[0]]
    for i in range(1, n_clusters):
        best_idx = -1
        best_min_dist = -1
        for j in idx_sorted[i:]:
            # Min distance to existing centers
            min_dist = min(abs(scores[j] - scores[c]) for c in centers_idx)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = j
        if best_idx >= 0:
            centers_idx.append(best_idx)
    
    centers = np.array([scores[i] for i in centers_idx])
    
    # Assign and update
    for _ in range(max_iter):
        # Assign each point to nearest center
        dists = np.abs(scores[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        
        # Update centers
        new_centers = centers.copy()
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = np.mean(scores[mask])
        centers = new_centers
    
    # Select best point from each cluster
    selected = []
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            cluster_indices = np.where(mask)[0]
            # Pick the one with highest score in this cluster
            best_in_cluster = cluster_indices[np.argmax(scores[cluster_indices])]
            selected.append(best_in_cluster)
    
    return np.array(selected)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if _time_remaining() < 5.0:
        # Emergency: return EI/cost quickly
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0) / (cost + eps)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure all inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition functions
    ei = _expected_improvement(mu, sigma, f_best)
    pi = _pi(mu, sigma, f_best)
    
    # UCB for minimization: lower is better, so we negate for "higher is better"
    ucb_val = _ucb(mu, sigma, f_best, beta=1.0)
    # Convert to "higher is better" by negating
    ucb_score = -ucb_val
    
    # Normalize each acquisition to [0, 1] for combination
    def normalize(a):
        a_min = np.min(a)
        a_max = np.max(a)
        if a_max - a_min < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - a_min) / (a_max - a_min)
    
    ei_norm = normalize(ei)
    pi_norm = normalize(pi)
    ucb_norm = normalize(ucb_score)
    
    # Combine: weighted sum, emphasizing EI and exploration
    combined = 0.5 * ei_norm + 0.3 * ucb_norm + 0.2 * pi_norm
    
    # Cost-aware: divide by cost
    # Use a soft cost penalty: divide by cost, but also consider the absolute EI
    # A point with high EI but high cost may not be worth it
    cost_aware = combined / cost
    
    # Also compute a pure EI/cost as fallback
    ei_cost = ei / cost
    
    # Blend: primarily cost-aware combined, but ensure EI/cost contributes
    final_score = 0.7 * cost_aware + 0.3 * (ei_cost / (np.max(ei_cost) + 1e-9))
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
