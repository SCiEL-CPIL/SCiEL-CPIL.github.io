import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0

def _check_time():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid approach: Expected Improvement (EI) as the primary score,
    with a small UCB exploration bonus to encourage exploration of high-uncertainty
    regions. The EI formulation is standard for minimization.
    """
    if _check_time():
        mu = np.asarray(mu, dtype=np.float64)
        return -mu.astype(np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Floor sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu(x), sigma(x)^2):
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Compute the normalized distance
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical overflow in exp
    z_clipped = np.clip(z, -10, 10)
    
    # Standard normal CDF: Phi(z) = 0.5 * erfc(-z/sqrt(2))
    # Use scipy if available, otherwise approximate
    try:
        from scipy.special import erf
        phi_cdf = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
        phi_pdf = np.exp(-0.5 * z_clipped**2) / np.sqrt(2.0 * np.pi)
    except ImportError:
        # Fallback: use numpy's approximation
        from math import sqrt, pi
        sqrt2 = np.sqrt(2.0)
        # Approximation for erf
        def erf_approx(x):
            # Abramowitz and Stegun approximation
            sign = np.sign(x)
            x = np.abs(x)
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            p = 0.3275911
            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
            return sign * y
        phi_cdf = 0.5 * (1.0 + erf_approx(z_clipped / sqrt2))
        phi_pdf = np.exp(-0.5 * z_clipped**2) / np.sqrt(2.0 * np.pi)
    
    # EI score
    ei_score = (f_best - mu) * phi_cdf + sigma_safe * phi_pdf
    
    # Add a small UCB exploration bonus to encourage exploration
    # The bonus is scaled to be small relative to EI when EI is large
    ucb_bonus = 0.1 * sigma_safe
    
    # Final score: EI + small exploration bonus
    score = ei_score + ucb_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
