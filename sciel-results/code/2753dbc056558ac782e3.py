# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Uncertainty-Weighted Improvement (UWI).
    
    Uses a local baseline estimated via nearest-neighbor averaging in the (mu, sigma)
    feature space to target boundaries of explored basins. The UWI score rewards
    candidates where the GP posterior variance is high relative to the local prediction.
    
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
    
    # Build a 2D feature space from (mu, sigma) for nearest-neighbor baseline estimation
    # Normalize each dimension to [0, 1] for consistent distance scaling
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Build KD-tree on normalized (mu, sigma) space
    features = np.column_stack([mu_norm, sigma_norm])
    tree = cKDTree(features)
    
    # For each candidate, find K nearest neighbors and compute local baseline
    # as the weighted average of their mu values (lower is better for minimization)
    K = 20  # number of neighbors
    
    # Query K nearest neighbors for all points
    distances, indices = tree.query(features, k=K)
    
    # Compute local baseline as weighted average of neighbor mu values
    # Weights are based on distance (closer neighbors get higher weight)
    # Use inverse distance weighting
    inv_distances = 1.0 / (distances + 1e-9)
    weights = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)
    
    # Get mu values of neighbors
    neighbor_mu = mu[indices]
    
    # Weighted average of neighbor mu values
    local_baseline = np.sum(weights * neighbor_mu, axis=1)
    
    # Blend local baseline with global f_best to ensure we don't miss global optimum
    # Use 60% local, 40% global
    blended_baseline = 0.6 * local_baseline + 0.4 * f_best
    
    # Compute EI with the blended baseline
    z = (blended_baseline - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (blended_baseline - mu) * cdf + sigma * pdf
    
    # Uncertainty weighting: reward candidates where sigma is high relative to
    # the scale of the expected improvement
    uncertainty_weight = sigma / (sigma + np.abs(ei) + 1e-9)
    
    # Final score: EI times (1 + uncertainty weight), divided by cost
    scores = ei * (1.0 + uncertainty_weight) / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
