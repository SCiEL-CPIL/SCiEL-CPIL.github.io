import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimization.
    
    For the 1D sphere function with a tight GP length scale (0.2),
    the posterior is well-localized. Pure EI provides the right balance:
    - Exploits low-mu regions (close to the minimum)
    - Naturally accounts for uncertainty via sigma
    - No need for ad-hoc exploration/exploitation scheduling
    
    The GP's tight length scale means that exploration is implicitly
    handled by the prior: regions far from observed points have high sigma,
    which EI will naturally explore when f_best is still high.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    #       = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negativity (EI should always be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
