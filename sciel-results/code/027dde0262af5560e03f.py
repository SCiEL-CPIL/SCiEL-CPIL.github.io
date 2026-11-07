"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration based on the uncertainty profile.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (n,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB with adaptive beta
    # Use a larger beta for exploration when uncertainty is high
    # beta is chosen to balance exploitation and exploration
    beta = 2.0
    
    ucb = mu + beta * sigma
    
    # For minimization, we want to minimize mu and maximize uncertainty
    # So for UCB in minimization context, we use: -mu + beta * sigma
    # Actually, for minimization, we want low mu values
    # The UCB for minimization should encourage exploring regions with high uncertainty
    # while also favoring regions with low predicted values
    
    # Let's use a hybrid: primarily EI but with a small UCB component
    # to encourage exploration in uncertain regions
    
    # Normalize EI and UCB components to combine them
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # For minimization, lower mu is better, so -mu favors low predictions
    # But we also want to explore high uncertainty regions
    # Use: score = -mu + beta * sigma (this encourages both low predictions and high uncertainty)
    
    # Actually, let's think about this more carefully.
    # For minimization:
    # - We want to find points where mu is low (good predicted value)
    # - We want to explore points where sigma is high (uncertain regions)
    # - EI already captures both of these in an optimal way
    
    # The standard EI is already a strong acquisition function.
    # Let's try a modified version that adds a small exploration bonus.
    
    # Compute a simple exploration bonus based on sigma
    # This encourages exploring regions with higher uncertainty
    exploration_bonus = sigma
    
    # Combine EI and exploration bonus
    # Use a weighted sum
    weight_ei = 0.9
    weight_explore = 0.1
    
    scores = weight_ei * ei + weight_explore * exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
