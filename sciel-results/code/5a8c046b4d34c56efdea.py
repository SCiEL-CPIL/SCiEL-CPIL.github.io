import numpy as np
from scipy.stats import norm

# Global state to track the number of acquisition calls
# This allows for time-adaptive exploration parameters
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Time-Adaptive GP-UCB + Expected Improvement.
    
    This function computes scores for candidate points using a combination of
    Expected Improvement (EI) and a UCB-style exploration term. The exploration
    parameter beta decays over time to shift focus from broad exploration to
    local exploitation as the search progresses.
    
    Parameters:
    mu: np.ndarray, shape (N,)
        Predicted mean of the objective function.
    sigma: np.ndarray, shape (N,)
        Predicted standard deviation of the objective function.
    f_best: float
        The best (lowest) objective value observed so far.
        
    Returns:
    np.ndarray, shape (N,)
        Acquisition scores. Higher scores indicate more promising points.
    """
    global _call_count
    _call_count += 1
    t = _call_count
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clip sigma to a small positive value to avoid division by zero
    # and numerical instability.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Exploration Parameter (Beta)
    # Start with high exploration (beta=8.0) to escape local minima and find
    # the global region, then decay to low exploration (beta~0.5) to refine.
    # The decay is exponential with a time constant of ~10 steps.
    # Total steps are ~25, so this ensures a good balance.
    beta = 8.0 * np.exp(-t / 10.0) + 0.5
    
    # Total Score: EI + Exploration Term
    # Higher scores are preferred.
    scores = ei + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
