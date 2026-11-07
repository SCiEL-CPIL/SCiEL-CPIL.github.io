"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Use a hybrid acquisition function combining Expected Improvement (EI)
    and a UCB-like exploration term to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher is better.
    """
    # Clip sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) component
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) component
    # Beta controls exploration-exploitation tradeoff
    # For early optimization, we want more exploration
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Since we're minimizing, we want low mu and high sigma (uncertainty)
    # For UCB in minimization context, we use: -mu + beta * sigma
    # But actually, standard UCB for minimization is: mu - beta * sigma (lower is better)
    # Wait, let me reconsider. In Bayesian optimization for minimization:
    # - EI is already set up for minimization (f_best - mu)
    # - For UCB in minimization, we want to explore where uncertainty is high
    #   and where the mean is low. A common approach is to use:
    #   ucb_min = mu - beta * sigma  (lower values are more promising)
    # But since we're returning scores where HIGHER is better to evaluate next,
    # we need to negate this or use a different formulation.
    
    # Let's use a simpler approach: combine EI with a weighted uncertainty bonus
    # The EI already captures exploitation. We add a term that rewards high sigma
    # to encourage exploration.
    
    # Normalize EI to have zero mean and unit variance for stable combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    # Uncertainty bonus: higher sigma means more exploration
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    sigma_norm = (sigma - sigma_mean) / sigma_std
    
    # Combine: primarily EI, with a smaller exploration bonus
    # The coefficient for exploration can be tuned; 0.5 is a reasonable starting point
    exploration_weight = 0.5
    scores = ei_norm + exploration_weight * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
