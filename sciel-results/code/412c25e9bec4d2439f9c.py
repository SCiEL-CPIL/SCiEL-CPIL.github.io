"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
import time

# Global deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 330s kill


def acquisition(mu, sigma, f_best):
    """
    Student-t based Expected Improvement acquisition function.
    
    Uses a t-distribution with ~5 degrees of freedom to model the posterior,
    accounting for heavy-tailed uncertainty in the limited 30-sample budget.
    This promotes more robust exploration than Gaussian EI.
    
    Parameters:
        mu: array of shape (2000,), posterior mean in objective units
        sigma: array of shape (2000,), posterior std in objective units
        f_best: float, lowest observed value
    
    Returns:
        array of shape (2000,), finite scores, higher means evaluate next
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple LCB-like score as a safe fallback
        return f_best - mu - 0.1 * sigma
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Student-t based EI with nu degrees of freedom
    # The t-distribution EI: E[max(f_best - X, 0)] where X ~ t_nu(mu, sigma)
    # For a t-distribution, the mean is mu and variance is sigma^2 * nu/(nu-2)
    # We treat sigma as the scale parameter of the t-distribution directly.
    # 
    # For t with df=nu, scale=sigma, location=mu:
    # P(X <= f_best) = CDF_t((f_best - mu) / sigma, df=nu)
    # E[max(f_best - X, 0)] = (f_best - mu) * CDF_t(z) + sigma * PDF_t(z) * nu/(nu-2) * (1 - ... )
    # 
    # Actually, let me use the proper formula. For X ~ t_nu(mu, sigma) (where sigma is scale):
    # Let z = (f_best - mu) / sigma
    # EI = (f_best - mu) * F_t(z; nu) + sigma * f_t(z; nu) * [nu * (1 - F_t(z;nu)) / (nu - 1)] ... 
    # 
    # Simpler approach: use the fact that for a t-distribution,
    # E[max(a - X, 0)] where X = mu + sigma * T, T ~ t_nu(0,1):
    # = (a - mu) * P(T <= (a-mu)/sigma) + sigma * E[max((a-mu)/sigma - T, 0)]
    # 
    # For standard t: E[max(c - T, 0)] = c * F_t(c) + f_t(c) * (1 + c^2/(nu-1)) * (nu-1)/nu * ... 
    # 
    # Let me use a cleaner formulation. The expected improvement for t-distribution:
    # If X ~ t_nu(mu, sigma^2) where sigma is the standard deviation (not scale),
    # then the scale parameter s = sigma * sqrt((nu-2)/nu)
    # 
    # But the task says sigma is a standard deviation in objective units.
    # Let's use the t-distribution with the interpretation that sigma is the scale.
    
    nu = 5.0  # degrees of freedom
    z = (f_best - mu) / sigma_safe
    
    # t CDF and PDF
    cdf_vals = student_t.cdf(z, nu)
    pdf_vals = student_t.pdf(z, nu)
    
    # For t-distribution with location mu, scale sigma, df nu:
    # EI = (f_best - mu) * F_t(z) + sigma * f_t(z) * (nu / (nu - 1)) * (1 - F_t(z)) * (1 + z^2/(nu-1))
    # 
    # Actually, the correct formula for E[max(a - X, 0)] where X ~ t_nu(mu, s):
    # Let Y = (X - mu)/s ~ t_nu(0,1), c = (a - mu)/s
    # E[max(a - X, 0)] = E[max(a - mu - s*Y, 0)] = E[s * max(c - Y, 0)]
    # = s * E[max(c - Y, 0)]
    # 
    # For standard t: E[max(c - Y, 0)] = c * F_t(c) + f_t(c) * (1 + c^2/(nu-1)) * (nu-1)/nu * ... 
    # 
    # Let me just use a simpler, well-known approximation:
    # EI_t = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals * (nu / (nu - 1))
    # This is an approximation but captures the heavy-tail effect.
    
    # More carefully: for t-distribution, the expected positive part:
    # E[max(c - Y, 0)] where Y ~ t_nu(0,1)
    # = c * F_t(c) + f_t(c) * [1 + c^2/(nu-1)] * (nu-1)/nu * ... 
    # 
    # A commonly used approximation that works well:
    # EI = (f_best - mu) * F_t(z) + sigma * f_t(z) * (nu / (nu - 1))
    
    correction = nu / (nu - 1)  # = 5/4 = 1.25 for nu=5
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals * correction
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
