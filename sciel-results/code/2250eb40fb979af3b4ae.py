import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive exploration.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard EI: improvement over f_best
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Numerical stability: clip z to avoid overflow in normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus to encourage sampling in uncertain regions
    # This helps escape local optima in the low-budget regime
    exploration_bonus = 0.1 * sigma
    
    scores = ei + exploration_bonus
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
