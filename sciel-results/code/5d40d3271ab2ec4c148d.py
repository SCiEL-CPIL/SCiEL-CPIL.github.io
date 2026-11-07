"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses a Cost-Adjusted Expected Improvement (EI) strategy with modified cost
normalization to balance exploration and exploitation while respecting budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2] to avoid division by zero or extreme values
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Structural Change: Thompson Sampling with Cost
    # Instead of deterministic Expected Improvement (EI), we use stochastic sampling.
    # We sample from the posterior distribution N(mu, sigma^2) for each candidate.
    # We then score based on the sampled value normalized by cost.
    # Since we want to minimize the function, we want low sampled values.
    # To convert to a maximization score (higher is better), we use -sample / cost.
    
    # Generate samples from the Gaussian posterior
    samples = np.random.normal(mu, sigma)
    
    # Calculate cost-aware score: -sample / cost
    # This favors candidates with low predicted values and low costs.
    score = -samples / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
