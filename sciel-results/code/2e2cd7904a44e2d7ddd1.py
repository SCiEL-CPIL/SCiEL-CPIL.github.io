import numpy as np
import time
from scipy.stats import norm

# Global state for deadline tracking
_start_time = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    This function implements a Thompson Sampling strategy for cost-aware Bayesian Optimization.
    Instead of using a deterministic acquisition function like Expected Improvement (EI),
    we sample a function value y from the posterior distribution N(mu, sigma^2) for each candidate.
    We then select the candidate that minimizes the ratio y / cost(x).
    To fit the evaluator's interface (where higher scores are preferred), we return the negative
    of this ratio: score = -y / cost.
    
    This approach naturally balances exploration and exploitation through the stochastic sampling
    and explicitly accounts for evaluation costs by prioritizing candidates with low predicted
    values relative to their cost.
    """
    # 1. Deadline Guard
    # If we are too close to the time limit, fall back to a fast, deterministic method
    # to ensure the program terminates gracefully.
    if time.time() - _start_time > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        
        # Fallback: Standard EI / Cost
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -50, 50)
        cdf_val = norm.cdf(z_clipped)
        pdf_val = norm.pdf(z_clipped)
        ei = (f_best - mu) * cdf_val + sigma * pdf_val
        scores = ei / cost_safe
        
        return np.where(np.isfinite(scores), scores, 0.0)

    # 2. Standard Thompson Sampling Path
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Sample from the posterior predictive distribution
    # y ~ N(mu, sigma^2)
    # Using np.random.normal for speed
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Calculate the cost-aware score
    # We want to minimize samples / cost.
    # The evaluator selects the maximum score.
    # Therefore, we return the negative of the ratio.
    scores = - (samples / cost_safe)
    
    # Ensure all scores are finite
    # Replace any non-finite values (inf, nan) with a very low score
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
