"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm, t as t_dist

# Track start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a Student-t Process (STP) approximation for heavier-tailed uncertainty,
    combined with Expected Improvement (EI) for exploitation.
    """
    # Check deadline - if we're close to timeout, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return -mu
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Student-t Process approximation:
    # The predictive distribution is approximated as a t-distribution with nu degrees of freedom.
    # For a t-distribution with nu DOF, mean mu and variance sigma^2,
    # the standard deviation is sigma, but the tails are heavier.
    # We can approximate the Expected Improvement under a t-distribution.
    
    # Let nu = 3 (common choice for STP)
    nu = 3.0
    
    # For a t-distribution with mean mu, std sigma, and nu DOF:
    # The CDF is t_dist.cdf(x, nu, loc=mu, scale=sigma)
    # The PDF is t_dist.pdf(x, nu, loc=mu, scale=sigma)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # Under t-distribution:
    # EI(x) = (f_best - mu) * P(f(x) < f_best) + E[f_best - f(x) | f(x) < f_best] * P(f(x) < f_best)
    #        = (f_best - mu) * CDF_t(f_best) - sigma * (nu + (f_best-mu)^2/sigma^2) / (nu-1) * PDF_t(f_best) ... 
    # Actually, the exact formula for t-EI is more complex. Let's use a simpler approximation:
    # We can use the fact that for large nu, t approaches normal. For small nu, we can scale the EI.
    
    # A practical approach: compute EI using the t-distribution CDF and PDF.
    # EI_t(x) = (f_best - mu) * CDF_t(f_best; mu, sigma, nu) + sigma * PDF_t(f_best; mu, sigma, nu) * (nu + z^2) / (nu - 1)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    cdf_t = t_dist.cdf(z, nu, loc=0, scale=1)  # Standard t CDF at z
    pdf_t = t_dist.pdf(z, nu, loc=0, scale=1)  # Standard t PDF at z
    
    # Correction factor for t-distribution vs normal
    # For t-distribution, the expected improvement has an extra term
    correction = (nu + z**2) / (nu - 1)
    
    ei_t = (f_best - mu) * cdf_t + sigma_safe * pdf_t * correction
    
    # Ensure non-negative EI
    ei_t = np.maximum(ei_t, 0.0)
    
    # UCB term for exploration with t-distribution
    # The quantile of the t-distribution for exploration
    # Use a high quantile, e.g., 95%
    beta_t = t_dist.ppf(0.95, nu, loc=0, scale=1)
    
    ucb_t = -mu + beta_t * sigma_safe
    
    # Combine EI and UCB
    # Normalize both
    ei_scale = np.std(ei_t) if np.std(ei_t) > 1e-9 else 1.0
    ucb_scale = np.std(ucb_t) if np.std(ucb_t) > 1e-9 else 1.0
    
    ei_norm = ei_t / ei_scale
    ucb_norm = ucb_t / ucb_scale
    
    # Weighted combination: 60% EI, 40% UCB (more exploration with STP)
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
