import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Probability of Improvement (PI) for minimization.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    # Replace non-finite sigmas with a small positive number
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1e-8)
    # Replace non-finite mus with f_best (neutral)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score: (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best
    # P(f(x) < f_best) = P((f(x) - mu) / sigma < (f_best - mu) / sigma) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf, though norm.cdf handles large values well
    # Clipping to [-10, 10] is safe as Phi(-10) ~ 0 and Phi(10) ~ 1
    z = np.clip(z, -10, 10)

    # Probability of Improvement
    pi = norm.cdf(z)

    # Ensure finite scores
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        # This is just a safety net, shouldn't trigger
        return -mu

    return pi
