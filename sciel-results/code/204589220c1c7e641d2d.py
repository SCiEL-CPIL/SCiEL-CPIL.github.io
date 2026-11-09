"""Rewritten program with Cost-Aware Hybrid EI/UCB Acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition function score for Bayesian Optimization.
    
    Args:
        mu: Array of predicted means (shape 2000,).
        sigma: Array of predicted standard deviations (shape 2000,).
        f_best: The lowest observed function value.
        cost: Array of evaluation costs for each candidate (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,), higher is better.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon to avoid division by zero and numerical instability
    eps = 1e-9
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) term for minimization
    # For minimization, we want low mu and high sigma (uncertainty).
    # The "optimistic" value is mu - kappa * sigma.
    # We want to maximize the score, so we use the negative of the optimistic value:
    # Score_component = -(mu - kappa * sigma) = -mu + kappa * sigma
    kappa = 1.5  # Exploration parameter
    ucb_term = -mu + kappa * sigma
    
    # 3. Combine EI and UCB terms
    # Blending EI (exploitation) with UCB (exploration)
    combined = ei + ucb_term
    
    # 4. Normalize by cost
    # Higher score means better acquisition value per unit cost
    scores = combined / (cost + eps)
    
    # 5. Ensure finite scores
    # Replace any non-finite values with a very low score (or 0) to avoid invalidity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
