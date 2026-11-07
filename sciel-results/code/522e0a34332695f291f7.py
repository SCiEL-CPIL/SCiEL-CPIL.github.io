"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    Uses a time-adaptive blend of Expected Improvement and a diversity-aware
    exploration term to balance exploitation and exploration.
    """
    t0 = time.time()
    deadline = 325.0  # safety margin before 330s kill

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Check time
    if time.time() - t0 > deadline - 2.0:
        return ei

    # Compute a simple diversity/exploration score based on how far each candidate
    # is from the current best-mu region. We use a proxy: points with higher sigma
    # and lower mu are more promising for exploration.
    # We'll create a score that rewards points that are in uncertain regions
    # but also have a chance of being good.
    
    # A simple approach: use the negative mean plus a scaled uncertainty
    # But to make it more robust, let's use a quantile-based approach.
    
    # Sort by EI to identify top candidates
    ei_sorted = np.sort(ei)
    
    # Determine a threshold: top 10% by EI are exploitation candidates
    top_10_pct_idx = int(0.10 * n)
    ei_threshold = ei_sorted[-top_10_pct_idx] if top_10_pct_idx > 0 else ei_sorted[-1]
    
    # For exploration, we want to consider points with high sigma that are not
    # already dominated by high EI points.
    
    # Create a combined score:
    # - For points with EI > threshold: use EI (exploitation)
    # - For others: use a blend of EI and sigma (exploration)
    
    # Normalize EI and sigma to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    if sigma_max - sigma_min > 1e-12:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Blend: 0.8 * EI_norm + 0.2 * sigma_norm for all points
    # This encourages exploration in uncertain regions while still favoring
    # points with high expected improvement.
    
    # Check time
    if time.time() - t0 > deadline - 1.0:
        return ei
    
    # Final score: blend of EI and a diversity term
    # The diversity term is sigma_norm, which rewards uncertain regions
    blend_weight = 0.2
    score = (1.0 - blend_weight) * ei_norm + blend_weight * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
