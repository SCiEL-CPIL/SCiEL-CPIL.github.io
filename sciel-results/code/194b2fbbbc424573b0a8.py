import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Improvement with adaptive exploration and quadratic cost penalty.
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # EI for minimization: improvement = f_best - f(x), we want E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Adaptive exploration: increase exploration when sigma is high relative to improvement
    # This helps escape local exploitation
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-12
    
    # Exploration bonus: proportional to sigma but scaled to be meaningful
    # Use a UCB-like term that's adaptive
    beta = 1.5 + 0.5 * (sigma_mean / (np.abs(f_best) + 1e-6))
    ucb_term = beta * sigma
    
    # Blend EI and UCB: EI handles exploitation well, UCB handles exploration
    # Weight them: 70% EI, 30% UCB normalized
    ei_max = np.max(ei) + 1e-12
    ucb_max = np.max(ucb_term) + 1e-12
    ei_norm = ei / ei_max
    ucb_norm = ucb_term / ucb_max
    
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Quadratic cost penalty: (1/cost)^2 to more aggressively penalize expensive points
    # Cost ranges from 1 to 2, so (1/cost)^2 ranges from 1.0 to 0.25
    cost_penalty = (1.0 / cost) ** 2
    
    score = combined * cost_penalty
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
