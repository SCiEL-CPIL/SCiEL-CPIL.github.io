import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    For each candidate, we sample a value from the posterior predictive distribution 
    and use the negative of that value as the score. The candidate with the highest 
    score (i.e., the lowest sampled value) is selected for evaluation.
    
    This approach naturally balances exploration and exploitation and is particularly 
    effective for multimodal functions like Styblinski-Tang.
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to EI if too close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n = len(mu)
    
    # Thompson Sampling:
    # Sample from N(mu, sigma^2) for each candidate.
    # We use a fixed seed for reproducibility within the evaluator's context if needed, 
    # but typically we want fresh randomness each call. 
    # The problem statement says "draws 2000 fresh uniform candidates... computes posterior... calls acquisition".
    # The acquisition function itself should provide the score.
    # Using np.random.default_rng() is good practice.
    
    rng = np.random.default_rng()
    
    # Ensure sigma is positive to avoid errors in normal distribution
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples
    samples = rng.normal(mu, sigma_safe)
    
    # The score is the negative of the sample, so that higher score means lower function value
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=-1e6, neginf=1e6)
    
    return scores
