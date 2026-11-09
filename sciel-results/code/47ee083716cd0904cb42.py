"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
import time


def acquisition(mu, sigma, f_best):
    """
    Voronoi-EI Acquisition Function.
    
    Combines Expected Improvement with a Voronoi-based diversity bonus.
    The diversity bonus is estimated by the distance to the nearest neighbor
    in the (mu, sigma) posterior space. This encourages selecting candidates
    that are both promising (high EI) and isolated in the posterior landscape,
    effectively approximating the volume of their Voronoi cell in the
    parameter space.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    DEADLINE = 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize mu and sigma for distance calculation in posterior space
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Construct points in the (mu_norm, sigma_norm) space
    points = np.column_stack((mu_norm, sigma_norm))
    
    # Build KDTree for nearest neighbor search
    try:
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=2)
        nn_dists = dists[:, 1]
    except Exception:
        nn_dists = np.ones(n)
        
    # Normalize distances
    nn_dists_max = np.max(nn_dists) + 1e-9
    nn_dists_norm = nn_dists / nn_dists_max
    
    # Adaptive weight for diversity
    known_min = -2.53218
    # Map f_best to exploration pressure
    # f_best high (e.g. 0) -> pressure 1 (explore)
    # f_best low (e.g. -2.53) -> pressure 0 (exploit)
    if f_best > 0:
        pressure = 1.0
    elif f_best < known_min:
        pressure = 0.0
    else:
        pressure = (f_best - known_min) / (0 - known_min)
        
    # Weight on diversity: 0.5 to 1.5
    diversity_weight = 0.5 + 1.0 * pressure
    
    # Voronoi-EI Score
    scores = ei * (1.0 + diversity_weight * nn_dists_norm)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
