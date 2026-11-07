# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Pure Expected Improvement (EI) acquisition function.
    
    This strategy relies on the fact that for 1D noiseless (or very low noise)
    functions like Styblinski-Tang, standard EI with a near-zero exploration
    parameter is highly effective. It directly maximizes the expected reduction
    in the objective value given the current GP posterior.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - current best observed value
        
    Returns:
        scores: array of shape (2000,) - EI scores (higher is better)
    """
    start_time = time.time()
    
    # Safety margin: return immediately if we are too close to the limit
    # though this calculation is O(N) and very fast for N=2000.
    if time.time() - start_time > 320:
        # Fallback: return a simple heuristic if time is critical (should never happen)
        return -mu + sigma

    # Ensure sigma is strictly positive to avoid division by zero
    # The GP noise is 1e-6, but numerical issues can make sigma 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of the standard normal distribution.
    
    # A small xi (exploration parameter) can be used. 
    # For noiseless problems, xi=0 is often optimal, but a tiny value like 1e-4 
    # or 1e-3 can prevent getting stuck if the GP is overconfident in a local min.
    # Given the inspirations state "Simple EI" scored 1.0, we use xi=0.
    # However, to be robust against potential numerical stagnation, we can use a 
    # very small xi relative to the scale of the function. 
    # Styblinski-Tang min is -39.16. Scale is ~100. 
    # Let's stick to xi=0 for pure exploitation as per the "Simple EI" inspiration.
    
    xi = 0.0
    
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all values are finite
    # If any NaNs or Infs appear, replace them with 0 (worst case: no improvement expected)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
