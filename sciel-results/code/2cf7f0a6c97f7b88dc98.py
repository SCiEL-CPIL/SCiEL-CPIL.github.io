"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0  # seconds, evaluator kills at 330s
_SAFETY_MARGIN = 5.0  # seconds

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Thompson Sampling with diversity penalty.
    
    Combines stochastic exploration (Thompson Sampling) with a batch-aware
    diversity penalty that discourages redundant evaluations in high-density
    regions. The penalty is based on pairwise kernel similarity among the
    top-K candidates.
    
    Args:
        mu: array of shape (2000,), posterior mean
        sigma: array of shape (2000,), posterior standard deviation
        f_best: float, best observed value so far
    
    Returns:
        array of shape (2000,), acquisition scores (higher = better)
    """
    # Deadline guard: if we're running out of time, return a fast fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fast fallback: simple UCB
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.5 * sigma_safe
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)
    
    # Thompson Sampling: draw a sample from the posterior
    # Use a local RNG seeded from the current state for reproducibility
    # within a step, but different across steps
    seed_val = int(np.sum(mu[:10]) * 1e6 + np.sum(sigma[:10]) * 1e3 + f_best * 100)
    seed_val = seed_val % (2**32)
    rng = np.random.RandomState(seed_val)
    z = rng.randn(n)
    
    # Sampled function values
    sampled_f = mu + sigma_safe * z
    
    # Base score: negative sampled value (we minimize, evaluator picks max score)
    base_scores = -sampled_f
    
    # Identify top-K candidates for diversity penalty
    K = min(50, n)
    top_indices = np.argsort(base_scores)[-K:]
    
    # Compute a diversity penalty based on pairwise similarity
    # We use the fact that points with high sigma are likely in unexplored regions,
    # and points with similar mu/sigma patterns are likely close together.
    # A simple proxy: penalize points that are close in (mu, sigma) space to
    # other top-K candidates.
    
    # Normalize mu and sigma for distance computation
    mu_range = np.max(mu) - np.min(mu) + 1e-9
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe) + 1e-9
    
    mu_norm = (mu - np.min(mu)) / mu_range
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / sigma_range
    
    # Compute pairwise distances in normalized (mu, sigma) space for top-K
    top_mu_norm = mu_norm[top_indices]
    top_sigma_norm = sigma_norm[top_indices]
    
    # Pairwise squared distances
    # Shape: (K, K)
    dm = top_mu_norm[:, None] - top_mu_norm[None, :]
    ds = top_sigma_norm[:, None] - top_sigma_norm[None, :]
    dist_sq = dm**2 + ds**2
    
    # Compute a "density" score for each top-K candidate:
    # How many other top-K candidates are nearby?
    # Use a Gaussian kernel with bandwidth based on the median distance
    median_dist = np.median(dist_sq[dist_sq > 0]) if np.any(dist_sq > 0) else 1.0
    bandwidth = max(median_dist * 0.5, 1e-6)
    
    # Pairwise kernel values (similarity)
    kernel_vals = np.exp(-dist_sq / (2 * bandwidth))
    np.fill_diagonal(kernel_vals, 0.0)
    
    # Density score: sum of similarities to other top-K candidates
    density_scores = np.sum(kernel_vals, axis=1)
    
    # Map density scores back to all points
    diversity_penalty = np.zeros(n)
    diversity_penalty[top_indices] = density_scores
    
    # Normalize the penalty
    penalty_range = np.max(diversity_penalty) - np.min(diversity_penalty) + 1e-9
    diversity_penalty_norm = diversity_penalty / penalty_range
    
    # Combine: base score minus a penalty for being in a dense region
    # The penalty weight controls how much diversity is encouraged
    penalty_weight = 0.5
    final_scores = base_scores - penalty_weight * diversity_penalty_norm
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
