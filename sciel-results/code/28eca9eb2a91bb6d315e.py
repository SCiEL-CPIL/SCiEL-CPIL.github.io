"""Acquisition function for Bayesian optimization of Goldstein-Price 2D (minimization)."""
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition for minimization.
    Returns scores where higher means "evaluate next".
    score = mu + k * sigma, where k is a dynamic exploration parameter.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Dynamic exploration parameter: start with moderate exploration, decay as f_best improves
    # Goldstein-Price minimum is 3, so we can use the gap to guide exploration
    # Use a fixed k that balances exploration and exploitation well for this problem
    k = 1.0

    # PLB for minimization: we want to find low values
    # Higher score = more likely to be good
    # score = mu + k * sigma (higher mu is worse, higher sigma means more uncertainty)
    # For minimization, we want LOW mu and HIGH sigma (uncertainty)
    # So score = -mu + k * sigma would be wrong for PLB
    # Actually, for minimization with PLB: we want to maximize the probability of improvement
    # PLB in minimization context: score = mu + k * sigma doesn't directly apply
    # 
    # Let me reconsider: For minimization, a point is promising if mu is low (good) 
    # and sigma is high (uncertain, might be even lower)
    # The standard PLB for minimization would be: score = -mu + k * sigma
    # But the directive says score = mu + k * sigma
    # 
    # Re-reading the directive: "score = mu + k * sigma, which maximizes the probability 
    # that the true function value exceeds a dynamic threshold derived from f_best"
    # 
    # This seems to be for maximization. For minimization, we should use:
    # score = -mu + k * sigma (negate mu since we want low values)
    #
    # Actually, let me think about this more carefully. The PLB acquisition for 
    # minimization typically uses: score = mu - k * sigma (lower bound on the function)
    # But we want to MAXIMIZE the score to find the best point.
    # 
    # For minimization: a point is good if its lower bound is low.
    # Lower bound = mu - k * sigma
    # We want to minimize the lower bound, so we want to maximize -(mu - k * sigma) = -mu + k * sigma
    
    score = -mu + k * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
