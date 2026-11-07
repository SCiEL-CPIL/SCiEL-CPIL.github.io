"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) for minimization.
    
    Parameters:
    mu: (2000,) array of predicted means.
    sigma: (2000,) array of predicted standard deviations.
    f_best: float, the best observed value so far.
    cost: (2000,) array of evaluation costs for each candidate.
    
    Returns:
    (2000,) array of scores. Higher score means more likely to be selected.
    """
    # Ensure inputs are numpy arrays and float
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or very small costs
    cost_safe = np.maximum(cost, 1e-6)
    
    # Avoid negative sigma due to numerical errors
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Adaptive Beta: 
    # We don't have the total spent cost, but we can estimate progress by the 
    # relative uncertainty. However, a fixed beta is often more robust for 
    # small budgets. 
    # For minimization, we look at the Lower Confidence Bound: L(x) = mu - beta * sigma.
    # We want to minimize L(x). 
    # The score should be high for low L(x). 
    # Score = -L(x) / cost = (beta * sigma - mu) / cost.
    
    # A beta of 2.0 is a reasonable starting point for 3D problems with small budgets.
    # It encourages exploration early (high sigma) but exploits low mu later.
    beta = 2.0
    
    # Calculate the score
    # We want to maximize this score.
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
