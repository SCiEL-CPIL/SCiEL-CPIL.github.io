import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Distance-to-Best weighted Expected Improvement to encourage exploration
    of distant basins while still exploiting promising regions.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # Higher EI means more likely to improve upon f_best
    z = (f_best - mu) / sigma_safe
    z = np.where(np.isfinite(z), z, 0.0)
    
    # Compute EI using standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute distance to the point with minimum mu (proxy for best point)
    # Since we don't have coordinates, we use the index with minimum mu as reference
    min_mu_idx = np.argmin(mu)
    
    # We need to create a distance metric. Since no coordinates are passed,
    # we'll use a combination of mu and sigma to create a pseudo-distance
    # that rewards points far from the current best region
    # Use normalized distance in the (mu, sigma) space
    mu_centered = mu - mu[min_mu_idx]
    sigma_centered = sigma_safe - sigma_safe[min_mu_idx]
    
    # Pseudo-distance: combine mu and sigma differences
    pseudo_dist = np.sqrt(np.maximum(mu_centered**2, 0) + np.maximum(sigma_centered**2, 0))
    
    # Normalize distance to [0, 1] range
    max_dist = np.max(pseudo_dist) if np.max(pseudo_dist) > 0 else 1.0
    pseudo_dist_norm = pseudo_dist / max_dist
    
    # Distance weight: higher for distant points, creating repulsive force
    # Use inverse distance with a floor to avoid division by zero
    # weight = 1 / (1 + alpha * dist) where alpha controls strength
    alpha = 2.0
    dist_weight = 1.0 / (1.0 + alpha * pseudo_dist_norm)
    
    # Invert so that distant points get HIGHER weight (repulsive force)
    dist_weight = 1.0 - dist_weight
    
    # Combine EI with distance weight
    # Scale EI by distance weight to encourage exploration of distant regions
    scores = ei * (0.5 + dist_weight)
    
    # Add a small UCB component for additional exploration
    beta = 1.5
    ucb_component = -mu + beta * sigma_safe
    scores = 0.7 * scores + 0.3 * ucb_component
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
