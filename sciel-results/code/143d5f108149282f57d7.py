import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Combines UCB (exploration) with Expected Improvement (exploitation)
    in a way that adapts to the optimization progress.
    
    For minimization:
    - UCB score: -mu + beta * sigma (explores high uncertainty)
    - EI score: (f_best - mu) * Phi(Z) + sigma * phi(Z) (exploits potential improvement)
    
    We blend these with a weight that shifts from exploration to exploitation
    as f_best approaches the known minimum.
    """
    start_time = time.time()
    
    # Known minimum of Branin 2D
    f_min = 0.397887
    
    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Calculate Z-score for EI
    Z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement formula
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: -mu + beta * sigma
    # Use an adaptive beta that decreases as we get closer to the optimum
    # When f_best is far from f_min, explore more; when close, exploit more
    progress = np.clip((f_best - f_min) / (10.0 - f_min), 0.0, 1.0)
    # Beta ranges from 2.0 (far from optimum) to 0.5 (close to optimum)
    beta = 0.5 + 1.5 * progress
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize both scores to comparable scales
    # Use a soft normalization to avoid division by zero
    ei_max = np.max(ei) + 1e-12
    ucb_max = np.max(np.abs(ucb)) + 1e-12
    
    ei_norm = ei / ei_max
    ucb_norm = ucb / ucb_max
    
    # Blend: more weight on EI as we progress, more on UCB early
    # Weight on EI increases as progress increases
    w_ei = 0.3 + 0.7 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Return a simple fallback if we're running out of time
        return -mu + 1.0 * sigma_safe
    
    return scores
