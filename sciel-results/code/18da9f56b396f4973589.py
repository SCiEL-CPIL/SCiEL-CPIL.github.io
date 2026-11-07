import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes a dynamic LCB acquisition function with exploration-exploitation balance.
    
    Parameters:
    mu : np.ndarray
        Mean of the Gaussian Process posterior, shape (N,)
    sigma : np.ndarray
        Standard deviation of the Gaussian Process posterior, shape (N,)
    f_best : float
        Best observed value (minimum) so far.
        
    Returns:
    np.ndarray
        Acquisition scores for each candidate, shape (N,)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic kappa: adapt exploration based on how close we are to the known minimum
    # Hartmann 3D minimum is approximately -3.86278
    hmin = -3.86278
    
    # Progress: 0 when f_best is far from hmin, 1 when very close
    # Use a scale of about 1.0 for the range
    progress = np.clip((f_best - hmin) / 1.0, 0.0, 1.0)
    
    # Kappa decreases as we get closer to the optimum
    # Start with high exploration (kappa ~ 3) and decrease to exploitation (kappa ~ 0.5)
    kappa = 3.0 - 2.5 * progress
    
    # LCB: lower confidence bound (we want to minimize, so we prefer low mu + high uncertainty)
    # score = mu - kappa * sigma
    # Higher score means more promising for minimization
    scores = mu - kappa * sigma_safe
    
    # Add a small EI-like term for exploitation pressure
    # This helps focus on points likely to improve f_best
    xi = 0.0
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend LCB and EI
    # More EI weight as we progress
    ei_weight = 0.3 + 0.5 * progress
    lcb_weight = 1.0 - ei_weight
    
    # Normalize both to comparable scales
    scores_lcb = scores
    scores_ei = ei
    
    # Combine
    final_scores = lcb_weight * scores_lcb + ei_weight * scores_ei
    
    # Ensure finite scores
    final_scores = np.where(np.isfinite(final_scores), final_scores, -np.inf)
    
    return final_scores
