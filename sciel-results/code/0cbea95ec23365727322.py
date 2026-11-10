# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv


def _compute_L4_L2_ratio(coeffs):
    """Compute L4/L2 ratio for given coefficients."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre quadrature in cos(theta)
    mu, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(mu)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Compute the spherical harmonic
    f = np.zeros_like(THETA, dtype=np.float64)
    
    for i in range(2 * k + 1):
        m = i - k
        c = coeffs[i]
        if c == 0:
            continue
        
        # Compute associated Legendre function P_10^m(cos theta)
        # Note: lpmv includes Condon-Shortley phase
        P = lpmv(abs(m), k, np.cos(THETA))
        
        # Normalization N_m
        N_m = np.sqrt(21 / (4 * np.pi) * np.math.factorial(k - abs(m)) / np.math.factorial(k + abs(m)))
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2) * N_m * P * np.cos(m * PHI)
        else:
            Y = np.sqrt(2) * N_m * P * np.sin(abs(m) * PHI)
        
        f += c * Y
    
    # Compute norms
    # Surface element: dS = sin(theta) dtheta dphi = -d(mu) dphi
    # So integral = sum w_i * dphi * f^p * sin(theta_i)
    # But mu = cos(theta), so sin(theta) = sqrt(1 - mu^2)
    sin_theta = np.sqrt(1 - mu**2)
    dphi = 2 * np.pi / n_phi
    
    # L2 norm: (1/4pi) * integral |f|^2 dS
    f2 = f**2
    integral_L2 = np.sum(w[:, None] * sin_theta[:, None] * f2 * dphi)
    L2_norm = np.sqrt(integral_L2 / (4 * np.pi))
    
    # L4 norm: (1/4pi) * integral |f|^4 dS
    f4 = f**4
    integral_L4 = np.sum(w[:, None] * sin_theta[:, None] * f4 * dphi)
    L4_norm = np.sqrt(integral_L4 / (4 * np.pi))
    
    return L4_norm / L2_norm


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the optimal spherical harmonic of degree 10
    that minimizes the L4/L2 norm ratio.
    """
    k = 10
    
    # For axisymmetric functions, only m=0 contributes
    # The optimal solution is the zonal harmonic
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    coeffs[k] = 1.0
    
    return coeffs


# EVOLVE-BLOCK-END
