"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a combination of Expected Improvement (EI) with a carefully tuned
    exploration bonus. The key structural change is using a normalized EI
    that accounts for the scale of uncertainty, combined with a UCB-like
    exploration term that decays as we gain more information.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Normalized EI: scale EI by the typical improvement scale
    # This makes EI comparable across different stages of optimization
    # Use the median of positive EI values as a scale reference
    positive_ei = ei[ei > 1e-10]
    if len(positive_ei) > 0:
        ei_scale = np.median(positive_ei)
        if ei_scale < 1e-10:
            ei_scale = 1.0
        ei_normalized = ei / ei_scale
    else:
        ei_normalized = ei
        ei_scale = 1.0
    
    # Exploration bonus using UCB-like approach
    # beta controls the exploration-exploitation tradeoff
    # As sigma decreases overall (more information), we want less exploration
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Adaptive beta: start high for exploration, decrease as we learn
    # The scale of sigma relative to f_best tells us about our knowledge
    # If sigma is large relative to f_best, we're still exploring
    # If sigma is small, we should exploit more
    
    # Use a logistic-like decay based on the ratio of typical sigma to f_best
    # When sigma_mean / f_best is large (early), beta is large
    # When sigma_mean / f_best is small (late), beta is small
    ratio = sigma_mean / (abs(f_best) + 1.0)
    # Map ratio to beta: beta ranges from about 0.5 to 5.0
    beta = 0.5 + 4.5 * np.exp(-2.0 * ratio)
    
    # UCB exploration term
    # Use sigma directly, not normalized, to preserve scale
    exploration = beta * sigma_safe
    
    # Combine: use a weighted sum where EI gets more weight as we improve
    # The weight on EI increases as ei_scale decreases (we're closer to optimum)
    # When ei_scale is small, improvements are small, so we should exploit more
    
    # Weight for EI: higher when we're close to the optimum
    # Use a sigmoid-like function of ei_scale
    # When ei_scale is large (early), weight_ei is smaller (more exploration)
    # When ei_scale is small (late), weight_ei is larger (more exploitation)
    weight_ei = 1.0 + 2.0 * np.exp(-3.0 * ei_scale)
    
    # Normalize exploration to be comparable to EI
    # Scale exploration by ei_scale so both terms are on similar magnitude
    exploration_scaled = exploration * ei_scale
    
    # Final score: weighted combination
    scores = weight_ei * ei_normalized + exploration_scaled
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
