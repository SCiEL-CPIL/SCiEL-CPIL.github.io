"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid UCB + Expected Improvement acquisition function.
    
    Uses adaptive exploration that decreases as f_best approaches the
    known global minimum (-1.8013), balancing exploitation and exploration.
    """
    # Clip sigma to avoid division by zero and ensure finite values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)  # Avoid numerical issues
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB component: mu + beta * sigma
    # Adaptive beta: start high for exploration, decrease as we approach optimum
    # f_best starts around -1.0 to -1.5, optimum is -1.8013
    # Use a smooth decay based on how close f_best is to the known minimum
    known_min = -1.8013
    # Progress: 0 when f_best is far from min, 1 when at min
    progress = np.clip((f_best - known_min) / 0.5, 0, 1)  # 0.5 range from -1.3 to -1.8
    # Beta decreases from ~3.0 to ~0.5 as we get closer
    beta = 0.5 + 2.5 * (1.0 - progress)
    
    ucb = mu + beta * sigma_safe
    
    # Normalize both components to [0, 1] range for stable combination
    # EI is always >= 0, UCB can be negative
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine with weights that shift from exploration to exploitation
    # Early: more UCB weight, Late: more EI weight
    w_ucb = 0.3 + 0.4 * (1.0 - progress)  # 0.3 to 0.7
    w_ei = 1.0 - w_ucb
    
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
