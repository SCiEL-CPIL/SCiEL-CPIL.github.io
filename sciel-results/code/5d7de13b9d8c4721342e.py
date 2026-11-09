"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0

def _matern52_kernel(x1, x2, length_scale=0.2):
    """Compute Matern-5/2 kernel between two sets of 1D points."""
    # x1, x2: arrays of shape (n,) and (m,) representing coordinates in [0,1]
    # We need to infer coordinates from mu/sigma patterns - use a heuristic approach
    # Since we don't have explicit coordinates, we'll use the ordering/index as proxy
    # and assume uniform spacing over [-0.5, 0.5] mapped to [0, 1]
    n1 = len(x1)
    n2 = len(x2)
    # Map indices to [0,1] assuming uniform distribution
    coords1 = np.linspace(0, 1, n1)
    coords2 = np.linspace(0, 1, n2)
    diff = coords1[:, None] - coords2[None, :]
    r = np.abs(diff) / length_scale
    # Matern-5/2: (1 + sqrt(3)*r + 3*r^2/3) * exp(-sqrt(3)*r)
    sqrt3_r = np.sqrt(3.0) * r
    kernel = (1.0 + sqrt3_r + sqrt3_r**2 / 3.0) * np.exp(-sqrt3_r)
    return kernel

def acquisition(mu, sigma, f_best):
    """
    Batch-aware, diversity-penalized acquisition function.
    
    Combines Expected Improvement with a diversity penalty that discourages
    selecting candidates that are highly correlated (close in function space)
    to other high-scoring candidates.
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Step 1: Compute Expected Improvement scores
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -8.0, 8.0)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Step 2: Identify top-K candidates for diversity analysis
    n_candidates = len(mu)
    K = min(50, n_candidates)  # Top 50 candidates
    
    # Get indices of top-K by EI score
    top_k_idx = np.argsort(ei)[-K:][::-1]
    
    # Step 3: Compute pairwise correlation matrix among top-K
    # Using Matern-5/2 kernel as proxy for correlation in function space
    # Since we don't have explicit coordinates, we use index-based proxy
    # The key insight: candidates with similar mu/sigma patterns are likely
    # in similar regions of the search space
    
    # Compute a simple similarity based on normalized mu and sigma
    # Normalize mu and sigma to [0,1] range for comparison
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min + 1e-12
    sigma_range = sigma_max - sigma_min + 1e-12
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Compute pairwise distances in (mu_norm, sigma_norm) space for top-K
    top_mu_norm = mu_norm[top_k_idx]
    top_sigma_norm = sigma_norm[top_k_idx]
    
    # 2D distance
    diff_mu = top_mu_norm[:, None] - top_mu_norm[None, :]
    diff_sigma = top_sigma_norm[:, None] - top_sigma_norm[None, :]
    dist = np.sqrt(diff_mu**2 + diff_sigma**2)
    
    # Convert distance to correlation-like measure using Matern-5/2
    # Smaller distance -> higher correlation
    length_scale = 0.5  # Tuned for the normalized space
    r = dist / length_scale
    sqrt3_r = np.sqrt(3.0) * r
    corr_matrix = (1.0 + sqrt3_r + sqrt3_r**2 / 3.0) * np.exp(-sqrt3_r)
    
    # Step 4: For each candidate, compute max correlation with top-K
    # This encourages diversity by penalizing candidates that are similar
    # to already-high-scoring candidates
    
    # For all candidates, compute their distance to each top-K candidate
    # and find the maximum correlation
    
    # Efficient computation: for each candidate, compute distance to all top-K
    # and take the maximum correlation
    
    # Reshape for broadcasting
    all_mu_norm = mu_norm[:, None]  # (n, 1)
    all_sigma_norm = sigma_norm[:, None]  # (n, 1)
    
    top_mu_norm_t = top_mu_norm[None, :]  # (1, K)
    top_sigma_norm_t = top_sigma_norm[None, :]  # (1, K)
    
    diff_all_mu = all_mu_norm - top_mu_norm_t  # (n, K)
    diff_all_sigma = all_sigma_norm - top_sigma_norm_t  # (n, K)
    
    dist_all = np.sqrt(diff_all_mu**2 + diff_all_sigma**2)  # (n, K)
    
    r_all = dist_all / length_scale
    sqrt3_r_all = np.sqrt(3.0) * r_all
    corr_all = (1.0 + sqrt3_r_all + sqrt3_r_all**2 / 3.0) * np.exp(-sqrt3_r_all)  # (n, K)
    
    # Max correlation with any top-K candidate
    max_corr = np.max(corr_all, axis=1)  # (n,)
    
    # Step 5: Apply diversity penalty
    # Penalize candidates that have high correlation with top-K
    # Use a penalty that scales with the EI score to balance exploration/exploitation
    
    # Normalize EI to [0, 1] for penalty scaling
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    # Penalty: higher correlation -> higher penalty
    # Scale penalty by EI to only penalize high-scoring candidates more
    penalty_strength = 0.3  # Tunable parameter
    diversity_penalty = penalty_strength * max_corr * ei_norm
    
    # Final score: EI minus diversity penalty
    scores = ei - diversity_penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
