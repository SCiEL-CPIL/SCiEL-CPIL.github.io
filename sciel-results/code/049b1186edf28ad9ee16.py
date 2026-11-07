# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Cost-Aware Lower Confidence Bound (LCB).
    
    Unlike the Expected Improvement (EI) approach, LCB uses a linear combination
    of the predicted mean and standard deviation to balance exploitation and
    exploration. This approach is often more computationally stable and robust
    to scaling issues compared to EI, which involves integrating the Gaussian
    tail.
    
    The score is defined as:
    score = (f_best - mu + kappa * sigma) / cost
    
    Where:
    - f_best: Current best observed value.
    - mu: Predicted mean from the GP.
    - sigma: Predicted standard deviation from the GP.
    - kappa: Exploration parameter (typically 1.0).
    - cost: Cost of evaluating the point.
    
    We divide by cost (not cost^alpha) to provide a linear penalty for cost,
    which can be more effective in preventing the algorithm from ignoring
    high-cost regions entirely if the improvement potential is high enough.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Exploration parameter
    # kappa=1.0 is a standard choice for LCB. 
    # We might tune this, but 1.0 is a safe, distinct starting point from EI.
    kappa = 1.0
    
    # Numerator: Potential improvement + exploration bonus
    # If mu < f_best, (f_best - mu) is positive (exploitation).
    # kappa * sigma encourages exploration (high uncertainty).
    numerator = (f_best - mu) + kappa * sigma_safe
    
    # Denominator: Cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score: Linear combination divided by cost
    score = numerator / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # If all scores are -inf (shouldn't happen with safe inputs), return 0s
    if np.all(score == -np.inf):
        score = np.zeros_like(score)
        
    return score
# EVOLVE-BLOCK-END
