import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    Uses a Hard Cost Gate on Expected Improvement (EI).
    Candidates with cost below the median cost of the batch receive full EI score.
    Candidates above receive a heavily penalized score.
    This forces exploitation of the low-cost region until uncertainty is low.
    
    Parameters:
    mu (np.array): Mean predictions from the GP, shape (N,).
    sigma (np.array): Standard deviation predictions from the GP, shape (N,).
    f_best (float): The best function value observed so far.
    cost (np.array): Cost of evaluating each candidate point, shape (N,).
    
    Returns:
    np.array: Scores for each candidate, higher is better.
    """
    # Time guard: If we are running out of time, return a safe default
    if _time_exceeded():
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma near zero to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Hard Cost Gate
    # Determine a threshold based on the distribution of costs in the current batch.
    # Using the 50th percentile (median) creates a binary-like split.
    # This forces the search to prioritize the cheaper half of the candidate space.
    cost_threshold = np.median(cost)
    
    # Apply the gate:
    # If cost < threshold, use full EI.
    # If cost >= threshold, apply a strong penalty (0.01) to effectively disable them
    # unless no cheap candidates have significant EI.
    penalty = 0.01
    scores = np.where(cost < cost_threshold, ei, ei * penalty)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
