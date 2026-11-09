"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Hypar (EHypR) acquisition function.
    
    Maximizes E[max(f_best, f(x)) / f(x)] where f(x) ~ N(mu, sigma^2).
    
    This is equivalent to minimizing E[min(1, f(x)/f_best)].
    Let S = E[min(1, Y/f_best)].
    We return 1/S (with safeguards) to maximize the acquisition score.
    """
    if f_best <= 1e-8:
        # Fallback if we've essentially found the minimum
        # Use -mu to exploit low means
        return -mu

    c = f_best
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z_c = (c - mu) / sigma
    # Clip z_c to avoid numerical issues in pdf/cdf for very large/small values
    z_c = (c - mu) / sigma_safe
    z_c = np.clip(z_c, -10, 10)
    
    # Phi(z_c) and phi(z_c)
    Phi_z = norm.cdf(z_c)
    phi_z = norm.pdf(z_c)
    
    # S = E[min(1, Y/c)]
    # If S is negative, it implies the GP predicts Y < 0 is likely, which is "good" for minimization to 0.
    # We want high scores for these points.
    # 1/S would be negative. We clip S to a small positive value to make 1/S large positive.
    
    term1 = 1.0
    term2 = Phi_z * (mu / c - 1.0)
    term3 = (sigma_safe / c) * phi_z
    
    S = term1 + term2 - term3
    
    # We want to minimize S. So we maximize -S.
    # This avoids the singularity at S=0 and provides a smoother acquisition landscape.
    acq = -S
    acq = np.nan_to_num(acq, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return acq
# EVOLVE-BLOCK-END
