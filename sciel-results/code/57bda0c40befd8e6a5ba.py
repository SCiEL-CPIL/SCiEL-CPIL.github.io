import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses Probability of Improvement (PI) normalized by cost^0.5.
    This provides a smoother, more stable gradient for exploration in multimodal landscapes.
    
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
    
    # Probability of Improvement (for minimization)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    improvement = f_best - mu
    z = improvement / sigma
    pi = norm.cdf(z)
    
    # Cost Adjustment
    # Divide by cost^0.5 to balance exploration efficiency against computational budget
    # This is a softer penalty than dividing by cost, allowing more exploration
    cost_factor = np.sqrt(cost)
    
    # Score: PI / sqrt(cost)
    score = pi / cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
