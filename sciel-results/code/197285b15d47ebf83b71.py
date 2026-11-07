"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for the t-distribution based acquisition
# We pre-compute constants for the t-distribution with nu degrees of freedom
# The strategy is to use a Student's t-distribution based acquisition to better
# handle the heavy-tailed uncertainty in the early stages of the optimization.

_NU = 5  # Degrees of freedom for the t-distribution

def _t_pdf(x, nu):
    """
    Compute the PDF of the Student's t-distribution.
    Using the formula: f(x) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + x^2/nu)^(-(nu+1)/2)
    """
    # We use scipy.special for the gamma function to be precise, but we can also use a lookup or approximation.
    # However, since we are in a loop, it's better to use a fast approximation or pre-compute the constant.
    # Let's use scipy.special.gammaln for stability.
    from scipy.special import gammaln
    c = np.exp(gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0)) / np.sqrt(nu * np.pi)
    return c * (1 + x * x / nu) ** (-(nu + 1) / 2.0)

def _t_cdf(x, nu):
    """
    Compute the CDF of the Student's t-distribution.
    We can use the relationship with the Beta function or just use scipy.stats.t.cdf which is fast.
    """
    from scipy.stats import t
    return t.cdf(x, nu)

# Pre-compute the constant for the t-PDF to avoid recomputing gammaln every time
from scipy.special import gammaln
_T_CONST = np.exp(gammaln((_NU + 1) / 2.0) - gammaln(_NU / 2.0)) / np.sqrt(_NU * np.pi)

def _t_pdf_fast(x):
    """Fast version of t-PDF with pre-computed constant."""
    return _T_CONST * (1 + x * x / _NU) ** (-( _NU + 1) / 2.0)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Student's t-distribution based Expected Improvement.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed value so far (minimum)
        
    Returns:
        scores: Array of acquisition values, shape (2000,)
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardize
    z = (f_best - mu) / sigma
    
    # Expected Improvement with Student's t-distribution
    # The t-distribution is heavier-tailed than the Gaussian, which means
    # it assigns more probability to extreme values, promoting exploration.
    
    # We use the t-CDF and t-PDF
    # Note: The standard EI formula is E[max(0, f_best - f(x))]
    # For a t-distribution, this is:
    # (f_best - mu) * P(f(x) < f_best) + E[(f_best - f(x)) * 1(f(x) < f_best)]
    
    # The second term is more complex for t-distributions.
    # A common approximation is to use the t-distribution's properties.
    # Alternatively, we can use the fact that for large nu, it approaches the Gaussian case.
    
    # Let's use a simpler approach: use the t-distribution to compute the probability
    # and then use a modified EI formula.
    
    # Actually, a common approach is to use the "Expected Improvement" formula
    # but with the t-distribution's CDF and PDF.
    
    # The EI for a t-distribution can be computed as:
    # EI = (f_best - mu) * CDF_t(z) + sigma * PDF_t(z) * (nu / (nu - 1)) * (1 - z^2/nu) / (1 + z^2/nu)
    # This is an approximation. Let's use a more standard form.
    
    # A simpler and more robust approach is to use the "Probability of Improvement"
    # combined with the mean, or use a hybrid.
    
    # Let's use the standard EI formula but with t-distribution functions:
    cdf_t = _t_cdf(z, _NU)
    pdf_t = _t_pdf_fast(z)
    
    # The expected improvement for a t-distribution is:
    # EI = (f_best - mu) * CDF_t(z) + sigma * PDF_t(z) * (nu / (nu - 1)) * (1 - z^2/nu) / (1 + z^2/nu)
    # But this can be negative or behave oddly. Let's use a safer formulation.
    
    # Actually, the most common approach in practice is to use the "Expected Improvement"
    # with the t-distribution's CDF and a modified second term.
    
    # Let's use a simpler heuristic: combine the standard EI with a t-distribution-based
    # exploration term.
    
    # Standard EI
    ei_gaussian = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # T-distribution based exploration
    # The t-distribution has heavier tails, so we can use its PDF to boost exploration
    # in regions with high uncertainty.
    
    # A common approach is to use the "Probability of Improvement" with the t-distribution
    # and add a small exploration term.
    
    # Let's use the following:
    # score = (f_best - mu) * cdf_t + sigma * pdf_t * correction_factor
    
    # The correction factor accounts for the difference between t and Gaussian distributions.
    # For nu=5, the t-distribution has heavier tails, so we should explore more.
    
    # Let's use a simple combination:
    # score = 0.7 * ei_gaussian + 0.3 * (f_best - mu) * cdf_t + sigma * pdf_t
    
    # Actually, let's just use the t-distribution based EI directly.
    # The formula for EI with a t-distribution is:
    # EI = (f_best - mu) * CDF_t(z) + sigma * PDF_t(z) * (nu / (nu - 1)) * (1 - z^2/nu) / (1 + z^2/nu)
    
    # Let's compute this:
    correction = (_NU / (_NU - 1)) * (1 - z * z / _NU) / (1 + z * z / _NU)
    correction = np.maximum(correction, 0)  # Ensure non-negative
    
    ei_t = (f_best - mu) * cdf_t + sigma * pdf_t * correction
    
    # Ensure the scores are finite
    ei_t = np.where(np.isfinite(ei_t), ei_t, 0)
    
    return ei_t
# EVOLVE-BLOCK-END
