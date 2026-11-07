"""
Cost-aware acquisition function using a hybrid of Cost-Adjusted Probability of Improvement
(CA-POI) and a UCB-style exploration term.

This approach combines the exploitability of POI (which is bounded and stable) with
a small exploration component (UCB) to encourage visiting uncertain regions, all
normalized by the evaluation cost. This structural change aims to balance the
tendency of pure EI to be dominated by large absolute improvements (which can be
misleading in low-budget settings) with the need to explore.
"""
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores.
    
    Args:
        mu: Mean of the GP surrogate, shape (N,)
        sigma: Standard deviation of the GP surrogate, shape (N,)
        f_best: Current best observed function value (scalar)
        cost: Cost of evaluating each candidate, shape (N,)
        
    Returns:
        scores: Acquisition scores, shape (N,). Higher is better.
    """
    # Deadline guard: if we're close to the limit, return a safe fallback
    # The computation is vectorized and fast, but we add a guard as per directive.
    if time.time() - _start_time > _DEADLINE:
        # Return a neutral score to avoid hanging or crashing
        return np.zeros_like(mu, dtype=np.float64)
    
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # If mu < f_best, z > 0, high probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi(z)
    # This is a bounded [0, 1] value, making it stable and exploitative
    poi = norm.cdf(z)
    
    # Calculate UCB-style exploration term (for minimization: -mu + beta*sigma)
    # This term rewards low mu (good) and high sigma (uncertainty).
    # We use a small weight for UCB to keep POI dominant but allow some exploration.
    beta = 1.0
    ucb_term = -mu + beta * sigma_safe
    
    # Combine POI and UCB. 
    # Since POI is in [0,1] and UCB can be large, we scale UCB down significantly.
    # The scale of mu depends on the function. Shekel 4d values are around -10 to -1.
    # A small weight like 0.01 to 0.05 is often appropriate.
    # Let's try a hybrid: primarily POI, with a small UCB boost.
    w_poi = 1.0
    w_ucb = 0.05
    
    score = (w_poi * poi + w_ucb * ucb_term) / cost_safe
    
    # Ensure all scores are finite
    # Replace NaNs or Infs with 0.0
    scores = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
