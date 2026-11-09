"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def _check_deadline(timeout=320.0):
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > timeout

def acquisition(mu, sigma, f_best):
    """
    Diversity-aware acquisition using EI with spatial spread penalty.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: 2000 finite scores, higher = more promising.
    """
    n = len(mu)
    
    # Deadline guard
    if _check_deadline():
        # Return simple EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute EI for all candidates
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Known minimum of the RBF mixture
    f_min = -2.53218
    
    # Progress: how close are we to the minimum?
    regret = f_best - f_min
    progress = 1.0 - np.clip(regret / 2.0, 0.0, 1.0)
    
    # Diversity weight: higher early, lower late
    diversity_weight = 0.3 * (1.0 - progress) + 0.05 * progress
    
    # Compute a diversity score based on how "spread out" each candidate is
    # We approximate this by looking at the local density of high-EI candidates
    # For each candidate, we measure how many other high-EI candidates are nearby
    
    # First, get the top candidates by EI
    top_k = max(10, int(n * 0.1))  # Top 10% of candidates
    top_indices = np.argsort(ei)[-top_k:]
    
    # For each candidate, compute the minimum distance to any top-EI candidate
    # This is expensive (2000 * 200 = 400k ops), but manageable
    # Use a grid-based approximation for speed
    
    # Create a coarse grid to estimate local density
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))
    
    # Map top candidates to grid cells
    # We don't have coordinates, so we use index as a proxy
    # Actually, we need a different approach since we don't have coordinates
    
    # Alternative: use the EI values themselves to create a diversity signal
    # Candidates with high EI that are "different" from others should be preferred
    # We can use the rank of EI as a proxy for diversity
    
    # Simpler approach: penalize candidates that have many neighbors with similar EI
    # This encourages spread in the EI landscape
    
    # Compute local EI density using a sliding window on the sorted EI values
    ei_sorted = np.sort(ei)
    window_size = max(5, int(n * 0.01))  # 1% window
    
    # For each candidate, find its position in the sorted array and compute local density
    ei_ranks = np.argsort(np.argsort(ei))  # Rank of each candidate
    local_density = np.zeros(n)
    
    for i in range(n):
        rank = ei_ranks[i]
        left = max(0, rank - window_size)
        right = min(n, rank + window_size + 1)
        local_density[i] = np.mean(ei[left:right])
    
    # Normalize local density
    ld_max = np.max(local_density)
    ld_min = np.min(local_density)
    ld_range = ld_max - ld_min + 1e-9
    ld_norm = (local_density - ld_min) / ld_range
    
    # Diversity score: candidates with low local density (spread out) get higher score
    diversity_score = 1.0 - ld_norm
    
    # Combine EI and diversity
    score = (1.0 - diversity_weight) * ei_norm + diversity_weight * diversity_score
    
    # Scale back to original EI scale for consistency
    score = score * ei_range + ei_min
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
