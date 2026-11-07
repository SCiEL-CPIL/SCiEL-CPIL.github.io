"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline checking
_start_time = time.time()
_TIME_LIMIT = 300.0  # Seconds
_SAFETY_MARGIN = 5.0  # Seconds

def _check_deadline():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TIME_LIMIT - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) with a dynamic exploration parameter.
    
    Args:
        mu: (2000,) array of mean predictions.
        sigma: (2000,) array of standard deviation predictions.
        f_best: float, the lowest observed value.
        
    Returns:
        (2000,) array of scores. Higher scores indicate more desirable candidates.
    """
    # Check deadline. If we are running out of time, we should return a valid result immediately.
    # Since EI computation is fast, this is mostly a precaution.
    if _check_deadline():
        # Fallback to a simple, fast exploration strategy if time is critical
        # Return high scores for high uncertainty to force exploration
        return sigma.copy()

    # Ensure sigma is positive (numerical stability)
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu - xi) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu - xi) / sigma
    
    # Dynamic exploration parameter xi.
    # We want to explore more if f_best is far from the known minimum (-1.8013).
    # Let's set xi based on the current f_best.
    # If f_best is -1.0, we might want xi = 0.1.
    # If f_best is -1.7, we might want xi = 0.05.
    # A simple heuristic: xi = 0.1 is a solid default for 2D multimodal.
    # To be more aggressive as per inspirations, let's use xi = 0.2.
    xi = 0.2
    
    # Calculate Z
    z = (f_best - mu - xi) / sigma
    
    # Standard Normal CDF and PDF
    # Using scipy.stats.norm for accuracy and speed
    from scipy.stats import norm
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate EI
    ei = (f_best - mu - xi) * Phi_z + sigma * phi_z
    
    # Ensure scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Higher EI is better.
    return ei

# EVOLVE-BLOCK-END
