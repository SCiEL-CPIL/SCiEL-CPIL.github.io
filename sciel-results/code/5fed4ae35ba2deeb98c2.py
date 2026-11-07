import numpy as np
from scipy.stats import t, norm

def acquisition(mu, sigma, f_best):
    # Robust EI with nu=5
    nu = 5.0
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Scaling factor for t-distribution
    scale = sigma_safe * np.sqrt((nu - 2) / nu)
    
    # Standardized variable
    z = (f_best - mu) / scale
    
    # CDF of t-distribution
    F = t.cdf(z, df=nu)
    
    # Compute J(z) = integral from -inf to z of t * pdf(t) dt
    # J(z) is even, so we compute for abs(z)
    z_abs = np.abs(z)
    
    # Constants
    C_nu = t.pdf(0, df=nu) / (1 + 0/nu)**((nu+1)/2) # This is just t.pdf(0, nu)
    # Actually t.pdf(0, nu) = C_nu
    C_nu = t.pdf(0, df=nu)
    
    # E[|T|] for t-dist
    # E[|T|] = sqrt(nu) * Gamma(nu/2) / (sqrt(pi) * Gamma((nu-1)/2))
    # We can compute this numerically or use scipy.special.gamma
    from scipy.special import gamma
    E_abs_T = np.sqrt(nu) * gamma(nu / 2) / (np.sqrt(np.pi) * gamma((nu - 1) / 2))
    
    # Term for z > 0
    # J(z) = C_nu * (nu / (nu - 1)) * (1 - (1 + z^2/nu)^(-(nu-1)/2)) - 0.5 * E[|T|]
    term1 = C_nu * (nu / (nu - 1)) * (1 - (1 + z_abs**2 / nu)**(-(nu - 1) / 2))
    J = term1 - 0.5 * E_abs_T
    
    # EI = (f_best - mu) * F - scale * J(z)
    # Note: J(z) in the formula is the integral.
    # Our J computed above is for z_abs. Since J is even, J(z) = J(z_abs).
    EI = (f_best - mu) * F - scale * J
    
    # Ensure finite
    EI = np.nan_to_num(EI, nan=0.0, posinf=0.0, neginf=0.0)
    
    return EI
