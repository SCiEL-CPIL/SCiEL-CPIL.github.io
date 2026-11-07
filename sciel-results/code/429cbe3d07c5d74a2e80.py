"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition for minimization.
    
    For minimization, we want to select points with low predicted mean and high uncertainty.
    LCB = mu - beta * sigma
    
    We use an adaptive beta that increases as uncertainty grows relative to the current best,
    providing stronger exploration in high-uncertainty regions to escape local minima
    in the rugged Weierstrass landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Posterior mean predictions
    sigma : ndarray, shape (n,)
        Posterior standard deviation predictions
    f_best : float
        Best observed value so far
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher is better)
    """
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    sigma = np.maximum(sigma, 1e-10)
    
    # Scale-aware adaptive beta:
    # When uncertainty is large relative to the gap to best, explore more aggressively.
    # This helps escape local minima in the Weierstrass function.
    
    # Base exploration coefficient
    beta_base = 2.5
    
    # Adaptive component: increase beta when sigma is large relative to (f_best - mu)
    # This creates stronger exploration in regions with high uncertainty
    gap = f_best - mu  # Positive where mu < f_best (potential improvement)
    
    # Normalize uncertainty by the scale of the problem
    # Use a robust measure of the typical gap
    median_gap = np.median(gap)
    if median_gap < 1e-10:
        median_gap = 1e-10
    
    # Adaptive beta: higher when uncertainty is large relative to typical improvement
    # beta = beta_base * (1 + alpha * sigma / |median_gap|)
    alpha = 0.5
    adaptive_factor = 1.0 + alpha * sigma / np.abs(median_gap)
    beta = beta_base * adaptive_factor
    
    # Clip beta to prevent extreme values
    beta = np.clip(beta, 1.0, 10.0)
    
    # LCB for minimization: lower is better, so we want to maximize (mu - beta * sigma)
    # But since we return "higher meaning evaluate next", we return the negative of LCB
    # Actually, for minimization with LCB, we select the point with the lowest LCB value
    # LCB_i = mu_i - beta_i * sigma_i
    # We want to maximize this score, so we return -LCB = -mu + beta * sigma
    # Wait, let me reconsider.
    
    # For minimization:
    # - We want points where mu is low (good prediction)
    # - We want points where sigma is high (exploration)
    # - LCB = mu - beta * sigma gives a lower bound on the true value
    # - We select the point with the LOWEST LCB (most optimistic for minimization)
    # - To convert to "higher is better", we return -LCB = -mu + beta * sigma
    
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
