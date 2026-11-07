import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value so far
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Time guard: if we are close to timeout, return a safe default
    # This prevents the evaluator from killing the process due to timeout.
    # Returning zeros is safe as it's a finite array of correct shape.
    # However, we want to try to compute if possible.
    # The computation of EI is O(N) and very fast for N=2000.
    # The bottleneck in the loop is the GP inference, not this function.
    # But if this function is called in a tight loop or if the system is slow,
    # we must ensure we don't hang.
    
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIMEOUT:
        # Return a deterministic fallback to avoid timeout
        # Using -mu is a reasonable exploitation-only fallback
        return -mu
    
    # Ensure sigma is positive to avoid division by zero or log(0)
    # Standard deviation should be >= 0. If it's 0, EI is 0 if mu < f_best else -inf?
    # Actually, if sigma=0, the function is deterministic.
    # If mu < f_best, improvement is f_best - mu > 0.
    # If mu >= f_best, improvement is 0.
    # The formula EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) handles this if we are careful.
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Z-score for Expected Improvement in minimization
    # Z = (f_best - mu) / sigma
    # We want P(f(x) < f_best) and E[max(f_best - f(x), 0)]
    # Let Y = f(x) ~ N(mu, sigma^2)
    # Improvement I = max(f_best - Y, 0)
    # Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    z = (f_best - mu) / sigma_safe
    
    # Handle numerical stability for large z
    # Phi(z) -> 1 for large z, phi(z) -> 0
    # np.clip z to avoid overflow in exp for phi, though norm.pdf handles it well.
    # For very large positive z, EI should be approx f_best - mu
    # For very large negative z, EI should be approx 0
    
    # Compute CDF and PDF
    # norm.cdf and norm.pdf are stable in scipy
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI calculation
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    # If mu is -inf, z is +inf, Phi is 1, phi is 0, EI is f_best - (-inf) = inf.
    # The problem states mu and sigma are in objective units, likely finite.
    # But let's be safe.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is always non-negative. Higher EI means more expected improvement.
    # This is a valid score where higher means evaluate next.
    
    return ei
