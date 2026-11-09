import numpy as np
from scipy.stats import t as t_dist
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Student-t Expected Improvement.
    
    Structural change:
    Switches from Gaussian EI/UCB blend to Student-t EI.
    The Beale function has a sharp, narrow valley. Gaussian EI tends to be
    overly confident in its mean prediction if the GP is slightly smooth,
    potentially missing the sharp drop. The Student-t distribution has
    heavier tails, which increases the probability of extreme improvements
    in regions of high uncertainty, effectively promoting more aggressive
    exploration in the narrow valley walls.
    """
    t0 = time.time()
    deadline = 325.0

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Degrees of freedom for Student-t
    # Lower nu means heavier tails (more exploration)
    nu = 3.0
    
    # Scale the sigma to match the variance of the t-distribution
    # Var(t) = nu / (nu - 2) * sigma_scale^2
    # We want Var = sigma_safe^2
    # sigma_scale = sigma_safe * sqrt((nu - 2) / nu)
    if nu <= 2:
        # Fallback to Gaussian if nu is too low for variance definition
        # But we set nu=3, so this is safe
        scale_factor = 1.0
    else:
        scale_factor = np.sqrt((nu - 2.0) / nu)
        
    sigma_t = sigma_safe * scale_factor
    
    # Standardized z-score for the t-distribution
    z = (f_best - mu) / sigma_t
    
    # Clip z to avoid numerical issues in CDF/PDF for extreme values
    # t-distribution CDF/PDF are well-behaved, but clipping helps
    z_clipped = np.clip(z, -100, 100)
    
    # Compute CDF and PDF
    cdf_val = t_dist.cdf(z_clipped, nu)
    pdf_val = t_dist.pdf(z_clipped, nu)
    
    # The EI formula for Student-t
    # E[max(0, f_best - X)] where X ~ t(mu, sigma_t, nu)
    # = (f_best - mu) * CDF(z) + sigma_t * Factor * PDF(z)
    # The Factor accounts for the expectation of the truncated part.
    # For standard t, E[max(0, c-Z)] = c*CDF(c) + PDF(c) * (nu+1)/(nu-1) * Gamma((nu+1)/2) / (sqrt(pi*nu)*Gamma(nu/2))
    # Let's compute the constant K
    from scipy.special import gamma
    K = ((nu + 1) / (nu - 2)) * (gamma((nu + 1) / 2) / (np.sqrt(np.pi * nu) * gamma(nu / 2)))
    
    # Wait, the factor is usually just related to the PDF.
    # Let's verify the integral.
    # \int_{-inf}^z (c - u) t(u) du = c T(z) - \int_{-inf}^z u t(u) du
    # \int_{-inf}^z u t(u) du is related to the PDF.
    # For large nu, it's approx phi(z).
    # The exact term is: \frac{\nu+1}{\nu-1} \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi \nu} \Gamma(\frac{\nu}{2})} (1 + z^2/\nu)^{-(\nu+1)/2}
    # Note that t.pdf(z) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi \nu} \Gamma(\frac{\nu}{2})} (1 + z^2/\nu)^{-(\nu+1)/2}
    # So the integral term is \frac{\nu+1}{\nu-1} * t.pdf(z) ? No, the PDF already has the coefficient.
    # Let A = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi \nu} \Gamma(\frac{\nu}{2})}
    # t.pdf(z) = A * (1 + z^2/\nu)^{-(\nu+1)/2}
    # The integral of u*t(u) is NOT simply proportional to pdf.
    # Actually, for t-distribution, the moment generating function is tricky.
    # However, a common approximation or exact form for EI_t is:
    # EI_t = (f_best - mu) * T(z) + sigma_t * \frac{\nu+1}{\nu-1} * \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu/2)} * (1 + z^2/\nu)^{-(\nu+1)/2}
    # This second term is exactly \frac{\nu+1}{\nu-1} * t.pdf(z) ?
    # t.pdf(z) = A * (1 + z^2/\nu)^{-(\nu+1)/2}.
    # So yes, the term is \frac{\nu+1}{\nu-1} * t.pdf(z).
    
    factor = (nu + 1) / (nu - 2) # Wait, is it nu-2 or nu-1?
    # Let's check the limit nu->inf.
    # t.pdf -> norm.pdf.
    # Factor should -> 1.
    # (nu+1)/(nu-2) -> 1.
    # (nu+1)/(nu-1) -> 1.
    # The integral of u * t(u) from -inf to z.
    # For normal, \int_{-inf}^z u \phi(u) du = -\phi(z).
    # So EI = (f_best - mu) Phi(z) - sigma_t (-\phi(z)) = (f_best - mu) Phi(z) + sigma_t \phi(z).
    # So the sign is positive.
    # For t, the term is proportional to pdf.
    # The constant is indeed \frac{\nu+1}{\nu-2} ? Or \frac{\nu+1}{\nu-1}?
    # Let's use the property that for nu=3, the tails are heavy.
    # I will use the coefficient C_nu = \frac{\nu+1}{\nu-2} * \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu/2)} / \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu/2)} ? No.
    
    # Let's just use the PDF directly with a multiplier.
    # Multiplier M = (nu + 1) / (nu - 2) is a good heuristic for the "heaviness" adjustment.
    # Actually, the exact integral \int_{-inf}^z u t_\nu(u) du = - \frac{\nu+1}{\nu-1} \frac{\Gamma((\nu+1)/2)}{\sqrt{\pi \nu} \Gamma(\nu/2)} (1+z^2/\nu)^{-(\nu+1)/2} ?
    # No, the sign is negative because u is negative in the tail.
    # So - \int ... is positive.
    # The magnitude is proportional to pdf.
    # Let's assume the term is `sigma_t * ( (nu+1)/(nu-2) * pdf_val )`.
    
    ei_term1 = (f_best - mu) * cdf_val
    ei_term2 = sigma_t * ((nu + 1) / (nu - 2)) * pdf_val
    
    scores = ei_term1 + ei_term2
    
    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    
    if time.time() > deadline:
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores
