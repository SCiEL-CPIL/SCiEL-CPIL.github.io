"""Combined acquisition: Expected Improvement with adaptive exploration via dynamic beta-UCB blend."""
import numpy as np
from scipy.stats import norm

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement and UCB for minimization.
    
    Uses a blend of EI (exploitation-focused) and UCB (exploration-focused)
    with a dynamic mixing parameter based on f_best. Early on (high f_best),
    more weight on UCB for exploration; later (low f_best), more weight on EI
    for exploitation.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    diff = f_best - mu
    z = diff / sigma_safe
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB for minimization: -mu + beta * sigma
    # Dynamic beta: higher when f_best is large (early), lower when f_best is small (late)
    beta_min = 0.5
    beta_max = 2.0
    scale = 0.5
    beta = beta_min + (beta_max - beta_min) * np.exp(-f_best / scale)
    
    ucb = -mu + beta * sigma_safe
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Normalize EI and UCB to comparable scales
    # EI is always >= 0, UCB can be negative
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min + 1e-12
    ucb_range = ucb_max - ucb_min + 1e-12
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Dynamic mixing: w is weight on EI, (1-w) is weight on UCB
    # When f_best is large (early), w is small (more UCB/exploration)
    # When f_best is small (late), w is large (more EI/exploitation)
    # Use sigmoid-like transition
    w = 1.0 / (1.0 + np.exp(-(f_best - 0.1) * 5.0))
    # Clamp w to [0.2, 0.8] to avoid extreme dominance
    w = np.clip(w, 0.2, 0.8)
    
    scores = w * ei_norm + (1.0 - w) * ucb_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
