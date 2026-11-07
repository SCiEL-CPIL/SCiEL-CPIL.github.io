# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Uncertainty-Weighted Improvement (UWI)
    with a local baseline estimated from nearest neighbors in the GP posterior space.
    
    Instead of using the global f_best, we estimate a local reference value for each candidate
    by looking at the GP means of its nearest neighbors in the (mu, sigma) feature space.
    This encourages the search to improve upon the local landscape rather than just chasing
    the global best or pure uncertainty.
    
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
    
    # Normalize features for KD-tree to ensure equal scaling
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Build KD-tree on normalized (mu, sigma) space
    features = np.column_stack([mu_norm, sigma_norm])
    tree = cKDTree(features)
    
    # Number of neighbors for local baseline estimation
    K = 15
    
    # Query K nearest neighbors for all points
    # Note: cKDTree.query returns k nearest neighbors, including the point itself if k>1
    # and the point is in the dataset.
    distances, indices = tree.query(features, k=K)
    
    # Get mu values of neighbors
    neighbor_mu = mu[indices]
    
    # Estimate local baseline: use a low quantile (e.g., 20th percentile) of neighbor mu values.
    # This is robust to outliers and represents the "promising" part of the local neighborhood.
    # Using the minimum is too sensitive to a single outlier, while the mean is too conservative.
    local_baseline = np.percentile(neighbor_mu, 20, axis=1)
    
    # Compute EI with the local baseline
    z = (local_baseline - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (local_baseline - mu) * cdf + sigma * pdf
    
    # Uncertainty weighting:
    # We add a term proportional to sigma to encourage exploration (UCB-like).
    # The coefficient is scaled by the standard deviation of the means to be adaptive.
    mu_std = np.std(mu) + 1e-9
    gamma = 0.5 * mu_std
    
    uwi_scores = ei + gamma * sigma
    
    # Final score: UWI divided by cost
    scores = uwi_scores / cost
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
