"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Gaussian Process Thompson Sampling (GP-TS).
    
    Instead of computing a deterministic score like EI or UCB, we sample a
    single realization of the GP posterior. The candidate with the lowest
    sampled value is selected. This inherently balances exploration (high
    uncertainty regions) and exploitation (low mean regions) without
    requiring manual tuning of exploration coefficients.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), higher is better
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check for sigma to avoid division by zero or log of negative numbers
    # Although sigma should be positive, numerical issues can arise
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Sample a single value from the GP posterior
    # x_sample ~ N(mu, sigma^2)
    # We use a fixed random state? No, the task says "draws 2000 fresh uniform candidates".
    # The evaluator likely expects the acquisition to be stochastic or deterministic based on inputs.
    # If we use np.random, it will be stochastic.
    # GP-TS is typically implemented by sampling y ~ N(mu, sigma^2) and picking argmin y.
    # The task asks to return scores where "higher meaning evaluate next".
    # So we should return -y_sample.
    
    # Generate random normal noise
    # Using a standard normal generator
    z = np.random.standard_normal(mu.shape)
    
    # Sampled values
    y_sample = mu + sigma_safe * z
    
    # We want to minimize y_sample, so higher score is better -> return negative
    scores = -y_sample
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
