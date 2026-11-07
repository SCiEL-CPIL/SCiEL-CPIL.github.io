import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement (PoI) acquisition function.
    
    This strategy abandons the cost-normalized EI and heuristic kappa search.
    Instead, it uses the standard PoI: P(f(x) < f_best).
    
    PoI is robust, deterministic, and has been observed to perform well 
    in similar Bayesian optimization contexts. It focuses on the likelihood 
    of improvement rather than the magnitude, which can be more stable 
    in the early-to-mid stages of optimization when f_best is still improving.
    
    The cost is implicitly handled by the GP's predictive variance: 
    points far from observed data have high sigma, which reduces PoI 
    (since the probability of being below f_best drops as uncertainty grows 
    unless mu is very low). Points close to observed data have low sigma, 
    allowing PoI to be high if mu is low. This creates a natural balance 
    between exploration (high sigma regions with potentially low mu) 
    and exploitation (low sigma regions with low mu).
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP posterior
    sigma: array of shape (2000,) - Standard deviation of the GP posterior
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores for PoI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # PoI = Phi(z)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Using scipy.stats.norm.cdf for accuracy
    poi_scores = norm.cdf(z_clipped)
    
    # Ensure all values are finite
    poi_scores = np.nan_to_num(poi_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return poi_scores
