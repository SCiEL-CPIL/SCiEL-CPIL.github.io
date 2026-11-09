import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a blend of Expected Improvement (EI) and Uncertainty-Weighted Probability of Improvement (PI*Sigma).
    The weights adapt based on the current best value f_best.
    The final score is penalized by the evaluation cost.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (for minimization)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Uncertainty-weighted PI
    pi_sigma = pi * sigma
    
    # Adaptive Weighting
    # Normalize f_best to [0, 1] where 0 is best (0) and 1 is worst (~20)
    f_scale = 10.0
    f_norm = min(max(f_best / f_scale, 0.0), 1.0)
    
    # Weights:
    # When f_best is high (f_norm close to 1), we need more exploration.
    # When f_best is low (f_norm close to 0), we need more exploitation.
    w_ei = 0.4 + 0.5 * (1.0 - f_norm)
    w_pi_sigma = 0.6 - 0.5 * (1.0 - f_norm)
    
    # Base Score
    base_score = w_ei * ei + w_pi_sigma * pi_sigma
    
    # Cost Adjustment: divide by cost^0.5 for smoother cost normalization
    score = base_score / np.sqrt(cost)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
