"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Compute Expected Improvement (EI) for minimization.
    # EI(x) = E[max(f_best - f(x), 0)]
    # Given f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Return EI as the score. Higher EI means more expected improvement.
    # We do not add an exploration term because EI naturally balances exploration
    # (via sigma in the second term) and exploitation (via f_best - mu in the first term).
    # Adding a raw sigma term (as in the previous program) effectively turns it into
    # a noisy UCB which can lead to excessive exploration and suboptimal regret in
    # low-budget settings like 30 evaluations.
    
    return ei
# EVOLVE-BLOCK-END
