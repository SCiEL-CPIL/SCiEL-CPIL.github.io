# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a single sample from the posterior at each candidate
    # sample ~ N(mu, sigma^2)
    # For minimization, we want to find low values, so we score by -sample / cost
    # This stochastically balances exploration (high sigma) and exploitation (low mu)
    
    # Draw one sample per candidate
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # For minimization: lower sample value is better
    # Score = -sample / cost (higher score = more desirable)
    # We want to prefer points with low predicted function value and low cost
    
    # Use a cost-aware scoring: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-6)
    
    # Primary score: negative sample value divided by cost
    scores = -samples / cost_safe
    
    # Add a small exploration bonus for high uncertainty to ensure we explore
    # This helps escape local minima
    # Scale the exploration bonus relative to the sigma range
    sigma_mean = sigma.mean()
    sigma_std = sigma.std() if sigma.std() > 0 else 1.0
    
    # Exploration bonus: proportional to sigma, but bounded
    # Use a soft saturation to prevent extreme exploration
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * (sigma - sigma_mean) / (sigma_std + 1e-8)
    
    # Combine: primary TS score + small exploration adjustment
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
