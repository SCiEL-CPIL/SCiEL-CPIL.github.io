<<<<<<< SEARCH
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    We want to minimize f(x).
    The GP mean is mu, std dev is sigma.
    EI(x) = E[max(f_best - f(x), 0)]
    For a GP with mean mu(x) and std sigma(x):
    Let z = (mu(x) - f_best) / sigma(x)
    EI = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where Phi is the CDF and phi is the PDF of the standard normal.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (mu - f_best) / sigma_safe
    
    # For minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But note: if mu < f_best, z is negative, and we expect improvement
    # Let's derive properly:
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # Let Y = f_best - f(x) ~ N(f_best - mu, sigma^2)
    # EI = E[max(Y, 0)]
    # For Y ~ N(m, s^2), E[max(Y, 0)] = m*Phi(m/s) + s*phi(m/s)
    # Here m = f_best - mu, s = sigma
    # So EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    m = f_best - mu
    s = sigma_safe
    
    # z for the normal CDF/PDF
    z_val = m / s
    
    # Compute EI
    ei = m * norm.cdf(z_val) + s * norm.pdf(z_val)
    
    return ei
=======
import numpy as np
from scipy.special import erf, exp

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    
    m = f_best - mu
    s = sigma_safe
    
    z = m / s
    
    # Compute CDF and PDF using scipy.special for efficiency
    sqrt_2 = np.sqrt(2.0)
    cdf_z = 0.5 * (1.0 + erf(z / sqrt_2))
    pdf_z = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = m * cdf_z + s * pdf_z
    
    return ei
>>>>>>> REPLACE
