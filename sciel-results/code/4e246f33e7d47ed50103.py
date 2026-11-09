"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf, exp


def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float, the minimum value observed so far
    
    Returns:
    scores: array of shape (2000,), higher is better
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0 (potential improvement)
    # If mu > f_best, z < 0 (unlikely improvement)
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Standard Normal PDF: phi(z) = (1 / sqrt(2 * pi)) * exp(-0.5 * z^2)
    
    sqrt_2 = np.sqrt(2.0)
    cdf_z = 0.5 * (1.0 + erf(z / sqrt_2))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If z is very negative, cdf_z ~ 0, pdf_z ~ 0, EI ~ 0.
    # If z is very positive, cdf_z ~ 1, pdf_z ~ 0, EI ~ f_best - mu.
    
    improvement = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    # If any NaNs or Infs appear, replace with -inf (worst case) or 0
    # However, the math above should produce finite values for finite inputs.
    # If sigma is 0 and mu < f_best, z is inf, cdf is 1, pdf is 0, EI is f_best - mu.
    # If sigma is 0 and mu > f_best, z is -inf, cdf is 0, pdf is 0, EI is 0.
    # np.maximum(sigma, 1e-9) prevents inf z, but limits the impact.
    # Actually, if sigma is 0, z should be handled carefully.
    # With sigma_safe, z is finite.
    
    # Clip to ensure no negative scores if we want? No, EI can be negative?
    # If mu > f_best, z < 0.
    # (f_best - mu) is negative. cdf_z is small positive.
    # sigma * pdf_z is positive.
    # Usually EI is defined as max(EI, 0) for acquisition?
    # Strictly, EI is the expected *positive* improvement.
    # The formula (f_best - mu)Phi(z) + sigma phi(z) is always >= 0 for z.
    # Let's verify:
    # Let g(z) = -z Phi(z) + phi(z) for z<0?
    # Actually, the standard formula yields non-negative values.
    # If z is large negative, Phi(z) is tiny, phi(z) is tiny.
    # (f_best - mu) is negative.
    # The term (f_best - mu)Phi(z) is negative.
    # The term sigma phi(z) is positive.
    # It is a known property that EI >= 0.
    
    # To be safe against floating point errors making it slightly negative,
    # we can clip to 0, but it's usually not necessary and might mask issues.
    # However, "higher meaning evaluate next". Negative scores are fine as long as they are lower than positive ones.
    
    return improvement
# EVOLVE-BLOCK-END
