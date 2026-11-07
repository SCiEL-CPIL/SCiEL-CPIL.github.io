import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive UCB with a beta that scales with the
    uncertainty landscape. This structural change from the fixed EI+0.1*sigma
    hybrid to an adaptive UCB approach should better balance exploration and
    exploitation across the three seeds.
    
    The key insight: use beta that adapts to the relative uncertainty at each
    point, encouraging exploration where the GP is uncertain relative to the
    overall uncertainty landscape.
    """
    # Deadline guard
    if time.time() - _MODULE_START_TIME > 325:
        return np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle numerical issues
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Compute adaptive beta based on the uncertainty landscape
    # sigma_std / sigma_mean gives a measure of local uncertainty relative to global
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # If sigma is nearly constant, use a moderate beta; otherwise scale
    if sigma_mean > eps:
        # Adaptive beta: base of 1.5, scaled by the ratio of std to mean
        # When uncertainty is high relative to mean, explore more
        # When uncertainty is low, exploit more
        uncertainty_ratio = sigma_std / (sigma_mean + eps)
        # Clamp the ratio to avoid extreme values
        uncertainty_ratio = np.clip(uncertainty_ratio, 0.1, 3.0)
        beta = 1.5 * (1.0 + 0.5 * uncertainty_ratio)
    else:
        beta = 1.5
    
    # UCB for minimization: we want to find the lowest value
    # score = mu - beta * sigma
    # Higher score means more promising (lower expected value with uncertainty bonus)
    score = mu - beta * sigma_safe
    
    # Add a small EI-like term to focus on improvement over f_best
    # This helps in later stages when we have good estimates
    z = (f_best - mu) / sigma_safe
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Blend: primarily UCB, with a small EI component
    # The EI term has different scale, so normalize it
    ei_scale = np.std(ei)
    ucb_scale = np.std(score)
    
    if ei_scale > eps and ucb_scale > eps:
        # Normalize both to comparable scales
        ei_norm = ei / ei_scale
        ucb_norm = score / ucb_scale
        # Weight: 70% UCB, 30% EI for a balance
        score = 0.7 * ucb_norm + 0.3 * ei_norm
    else:
        score = mu - beta * sigma_safe
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
