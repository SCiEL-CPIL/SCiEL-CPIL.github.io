import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB)
    with dynamic kappa based on budget consumption.
    
    Parameters:
    mu (np.array): Mean of the Gaussian Process.
    sigma (np.array): Standard deviation of the Gaussian Process.
    f_best (float): Best observed function value so far.
    cost (np.array): Cost of evaluating each candidate point.
    
    Returns:
    np.array: Acquisition scores for each candidate point (higher is better).
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Estimate budget consumption: total cost spent / total budget (30)
    # Use the mean cost as a proxy for remaining budget
    mean_cost = np.mean(cost)
    # Total cost so far is not directly available, but we can estimate from f_best
    # If f_best is very low, we've likely spent more budget
    # Use a simple heuristic: kappa increases as f_best decreases
    # Normalize f_best to [0, 1] assuming griewank range [0, 1]
    f_best_norm = np.clip(f_best, 0, 1)
    
    # Dynamic kappa: start aggressive (high kappa) when f_best is high (early),
    # decrease to exploit (low kappa) when f_best is low (late)
    kappa = 1.0 + 3.0 * f_best_norm
    
    # Cost-Adjusted LCB: maximize (mu - kappa * sigma) / cost
    # For minimization, we want low mu and high sigma (uncertainty)
    # LCB = mu - kappa * sigma, we maximize -LCB = kappa * sigma - mu
    lcb_score = kappa * sigma - mu
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-6)
    score = lcb_score / cost_safe
    
    # Sanitize
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
