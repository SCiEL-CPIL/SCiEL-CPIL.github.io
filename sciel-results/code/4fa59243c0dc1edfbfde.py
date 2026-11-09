"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# We don't know the exact step count, but we know total evaluations is 30.
# We can track the number of calls to estimate progress.
# However, the function signature does not allow state persistence easily without globals.
# Since the evaluator runs 3 seeds, and each seed has a fresh loop, we can use a global counter
# or just rely on f_best to determine the phase.
# A simpler, robust approach: use a fixed xi that is known to work well for Ackley,
# or use a heuristic based on f_best.
# Ackley min is 0. f_best starts high.
# If f_best is close to 0, we should exploit (low xi).
# If f_best is high, we should explore (high xi).

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive xi.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviation predictions
    f_best: float, best observed value so far
    
    Returns:
    (2000,) array of acquisition scores (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive xi: 
    # Ackley function values are typically in [0, ~10].
    # If f_best is large, we are in early exploration. Use larger xi to encourage exploration.
    # If f_best is small, we are in late exploitation. Use smaller xi.
    # Heuristic: xi = 0.05 * f_best, capped at 0.5, floored at 0.01.
    # This ensures that as f_best -> 0, xi -> 0, shifting to pure exploitation.
    xi = np.clip(0.05 * f_best, 0.01, 0.5)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu - xi) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu - xi) / sigma
    # Phi is the standard normal CDF, phi is the standard normal PDF
    
    improvement = f_best - mu - xi
    Z = improvement / sigma
    
    # Compute CDF and PDF
    # Using scipy.stats for numerical stability
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    ei = improvement * Phi + sigma * phi
    
    # Ensure finite outputs
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # EI is non-negative by definition if we use the standard formula, 
    # but due to numerical issues or specific parameterizations, 
    # we ensure it's a valid score.
    # Higher EI means higher priority.
    
    return ei
# EVOLVE-BLOCK-END
