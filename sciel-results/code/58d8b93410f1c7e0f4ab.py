import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Deadline guard
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Bayesian expected utility
    approach with soft-thresholded improvement and uncertainty weighting.
    
    Parameters:
    mu: np.ndarray, shape (2000,), GP mean predictions.
    sigma: np.ndarray, shape (2000,), GP standard deviation predictions.
    f_best: float, best observed function value.
    cost: np.ndarray, shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: np.ndarray, shape (2000,), acquisition scores.
    """
    elapsed = _get_elapsed()
    
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    if elapsed > 320:
        z = (f_best - mu) / (sigma + 1e-9)
        pi = 0.5 * (1 + np.tanh(0.7978845608028654 * z))
        return pi / np.maximum(cost, 1e-9)
    
    # Convert to arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Soft-thresholded improvement probability
    # lambda controls how quickly the probability saturates
    lam = 1.0
    improvement = np.maximum(0.0, f_best - mu)
    p_imp = 1.0 - np.exp(-lam * improvement)
    
    # Confidence/exploration term based on uncertainty
    # gamma controls how strongly we reward high uncertainty
    gamma = 2.0
    conf = 1.0 - np.exp(-gamma * sigma)
    
    # Combine and divide by cost
    # Use a softer cost scaling (square root) to avoid over-penalizing expensive points
    # while still preferring cheaper evaluations
    cost_factor = np.sqrt(cost)
    
    score = (p_imp * conf) / cost_factor
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
