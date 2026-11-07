import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Kernelized Thompson Sampling (KTS).
    
    Strategy:
    1. Sample a value y_i from N(mu_i, sigma_i^2) for each candidate i.
    2. Calculate the potential improvement: imp_i = f_best - y_i.
    3. Normalize by cost to account for the variable evaluation cost: score_i = imp_i / cost_i.
    4. Return the scores. The evaluator picks the argmax.
    
    This is a structural change from UCB/EI:
    - UCB uses a deterministic bound (mu + beta*sigma).
    - EI uses an expectation over the improvement.
    - KTS samples from the posterior, which provides a natural, stochastic balance 
      between exploration (high sigma) and exploitation (low mu) that is often 
      more robust in short-horizon settings than tuned UCB parameters.
    """
    
    # Ensure inputs are valid
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety check for sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Safety check for cost (avoid division by zero)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Kernelized Thompson Sampling:
    # Draw a sample from the posterior predictive distribution for each candidate.
    # y_i ~ N(mu_i, sigma_i^2)
    # We use a fixed random seed? No, Thompson Sampling relies on randomness to explore.
    # However, if we want reproducibility or stability, we might consider deterministic
    # approximations, but standard KTS uses random draws.
    # Since the function is called repeatedly in a loop, the randomness helps escape local optima.
    
    # Draw samples
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Calculate potential improvement: f_best - sample
    # If sample < f_best, this is positive (good).
    # If sample > f_best, this is negative (bad).
    improvement = f_best - samples
    
    # Cost-aware scoring:
    # We want to maximize improvement per unit cost.
    scores = improvement / cost_safe
    
    # Handle any numerical issues (NaNs, Infs)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
