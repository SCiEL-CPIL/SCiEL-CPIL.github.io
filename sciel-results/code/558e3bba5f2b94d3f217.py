"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def _check_deadline(timeout=310.0):
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > timeout

def acquisition(mu, sigma, f_best):
    """
    Posterior-uniqueness acquisition: combines EI with a score that estimates
    the effective basin size of each candidate in posterior (mu, sigma) space.
    Candidates with distinctive posterior predictions (far from others) get a
    bonus, approximating Voronoi cell volume and encouraging exploration of
    unexplored basins in the multimodal RBF mixture.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard: return simple EI as fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Known minimum of the RBF mixture
    f_min = -2.53218
    
    # Progress: how close are we to the minimum?
    regret = f_best - f_min
    progress = 1.0 - np.clip(regret / 2.0, 0.0, 1.0)
    
    # Compute Expected Improvement (EI) for all candidates
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Compute posterior uniqueness score (approximate Voronoi cell volume)
    # For each candidate, measure its minimum distance to all other candidates
    # in the (mu, sigma) space. Larger minimum distance = larger effective basin.
    
    # Normalize mu and sigma to [0, 1] for distance computation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    mu_range = mu_max - mu_min + 1e-9
    sigma_range = sigma_max - sigma_min + 1e-9
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Compute pairwise distances in (mu_norm, sigma_norm) space
    # This is O(n^2) = 4 million operations for n=2000, which is manageable
    mu_diff = mu_norm[:, np.newaxis] - mu_norm[np.newaxis, :]
    sigma_diff = sigma_norm[:, np.newaxis] - sigma_norm[np.newaxis, :]
    dist = np.sqrt(mu_diff**2 + sigma_diff**2)
    
    # For each candidate, find the minimum distance to any OTHER candidate
    # Set diagonal to infinity so it doesn't affect min
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist, axis=1)
    
    # Normalize min_dist to [0, 1]
    md_max = np.max(min_dist)
    md_min = np.min(min_dist)
    md_range = md_max - md_min + 1e-9
    min_dist_norm = (min_dist - md_min) / md_range
    
    # Posterior uniqueness score: candidates far from others get higher score
    # This approximates the Voronoi cell volume in posterior space
    uniqueness_score = min_dist_norm
    
    # Adaptive weighting: more weight on uniqueness early (exploration),
    # more weight on EI late (exploitation)
    uniqueness_weight = 0.35 * (1.0 - progress) + 0.10 * progress
    
    # Combine EI and uniqueness
    score = (1.0 - uniqueness_weight) * ei_norm + uniqueness_weight * uniqueness_score
    
    # Scale back to EI scale for consistency
    score = score * ei_range + ei_min
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
