"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid rational model: Linear polynomial + 2 Gaussian Radial Basis Functions.
    
    y(x) = p0 + p1*x + sum_{i=1}^{2} w_i * exp(-(x - c_i)^2 / sigma^2)
    
    This model combines a global linear trend with local Gaussian bumps.
    The Gaussians allow capturing non-linear features, local extrema, or
    sharp transitions that a pure polynomial might miss, while the linear
    part handles the overall trend. The parameters are linear in the basis
    functions, which helps BFGS convergence, although the centers c_i
    introduce non-linearity.
    
    Parameters (10 total):
    params[0]: p0 (intercept)
    params[1]: p1 (slope)
    params[2]: p2 (quadratic coeff)
    params[3]: p3 (cubic coeff)
    params[4]: w1 (weight for gaussian 1)
    params[5]: c1 (center for gaussian 1)
    params[6]: w2 (weight for gaussian 2)
    params[7]: c2 (center for gaussian 2)
    params[8]: sigma1 (width for gaussian 1)
    params[9]: sigma2 (width for gaussian 2)
    
    Note: The centers and sigmas are optimized by BFGS, making this a non-linear
    least squares problem. To improve stability, we clip sigma to a minimum value
    and centers to the domain [-2, 2] (with a small margin).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack parameters
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    w1, c1, w2, c2 = params[4], params[5], params[6], params[7]
    sigma1, sigma2 = params[8], params[9]
    
    # Polynomial part: a0 + a1*x + a2*x^2 + a3*x^3
    poly = p0 + x * (p1 + x * (p2 + x * p3))
    
    # Gaussian part
    # Ensure sigmas are positive and not too small to avoid numerical issues
    # A minimum sigma of 0.05 allows for fairly sharp peaks without dividing by near-zero
    s1 = max(abs(sigma1), 0.05)
    s2 = max(abs(sigma2), 0.05)
    
    # Clip centers to a reasonable range to keep the optimization stable
    # The domain is [-2, 2], so centers far outside are unlikely to be useful
    c1_clipped = np.clip(c1, -2.5, 2.5)
    c2_clipped = np.clip(c2, -2.5, 2.5)
    
    g1 = np.exp(-0.5 * ((x - c1_clipped) / s1) ** 2)
    g2 = np.exp(-0.5 * ((x - c2_clipped) / s2) ** 2)
    
    result = poly + w1 * g1 + w2 * g2
    
    return result
# EVOLVE-BLOCK-END
