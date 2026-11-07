"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and UCB.
    
    The current best program (score 1.4716) uses a blend of 80% EI and 20% UCB 
    with beta=0.5. We hypothesize that a slightly stronger exploration component 
    (higher beta and higher weight) might improve the score by helping escape 
    local minima in the Branin function more effectively, while still maintaining 
    the strong exploitation signal of EI.
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute UCB score for minimization
    # We want to maximize the potential for a low function value.
    # Lower bound estimate: mu - beta * sigma
    # Score to maximize: -(mu - beta * sigma) = -mu + beta * sigma
    # We increase beta from 0.5 to 1.0 to make the exploration component more distinct.
    beta = 1.0
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both components to have zero mean and unit variance
    # This ensures that the weights w_ei and w_ucb directly control the relative importance.
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-10
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # 4. Blend EI and UCB
    # We increase the UCB weight from 0.2 to 0.3 to allow for more exploration.
    w_ei = 0.7
    w_ucb = 0.3
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
