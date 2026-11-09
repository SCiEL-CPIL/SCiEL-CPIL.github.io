"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: The evaluator kills the program after 330s.
# We track the start time of the first call to enforce a global budget if needed,
# though individual calls should be sub-millisecond.
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses standard Expected Improvement (EI) for minimization.
    This approach relies on the GP posterior (mu, sigma) to balance
    exploration and exploitation naturally without heuristic blending.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: if we are close to the 330s limit, return a safe fallback
    # to avoid timeout. In practice, EI is fast, but this guards against
    # pathological cases or slow environments.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a flat score to prevent further expensive processing if this
        # were part of a larger internal loop (though the contract says the
        # evaluator handles the loop). For this specific contract, we just
        # ensure we return finite values quickly.
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values to prevent NaN/Inf propagation
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
    # f_best is the minimum observed value. For Ackley, min is 0.
    # If f_best is negative (shouldn't happen for Ackley), clamp to 0 or small positive.
    f_best = max(f_best, 0.0)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = E[max(0, f_best - f(x))]
    # Let Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    z = (f_best - mu) / sigma
    
    # Use standard normal CDF and PDF
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
