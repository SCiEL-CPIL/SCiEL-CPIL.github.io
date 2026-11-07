"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses an adaptive Lower Confidence Bound (LCB) strategy.
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian Process prediction, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP prediction, shape (2000,).
    f_best (float): The lowest observed objective value so far.
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,). Higher is better.
    """
    # Ensure f_best is a scalar float
    if hasattr(f_best, '__len__'):
        fb = float(np.min(f_best))
    else:
        fb = float(f_best)
        
    # Adaptive beta for LCB
    # We want to explore more (higher beta) when we are far from the minimum (high f_best)
    # and exploit more (lower beta) when we are close to the minimum (low f_best)
    beta = 1.0 + 0.5 * fb
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB for minimization is mu - beta * sigma
    # We want to select the point with the lowest LCB.
    # Since the acquisition function returns scores where higher is better,
    # we return the negative of the LCB, normalized by cost.
    # score = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    score = (beta * sigma - mu) / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
