"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a smooth elastic-plastic model with temperature dependence.
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Unpack parameters
    E = params[0]           # Young's modulus (elastic stiffness)
    eps_y0 = params[1]      # Base yield strain
    temp_yield = params[2]  # Temperature coefficient for yield strain
    H0 = params[3]          # Base hardening modulus
    temp_hard = params[4]   # Temperature coefficient for hardening
    sigma_y0 = params[5]    # Base yield stress
    temp_ystr = params[6]   # Temperature coefficient for yield stress
    n_hard = params[7]      # Hardening exponent (for power-law)
    smooth_width = params[8]  # Width of smooth yield transition
    sigma_ref = params[9]    # Reference stress scale
    
    # Temperature-dependent yield strain
    eps_y = eps_y0 * (1.0 - temp_yield * T)
    eps_y = np.maximum(eps_y, 1e-6)  # Ensure positive yield strain
    
    # Temperature-dependent yield stress
    sigma_y = sigma_y0 * (1.0 - temp_ystr * T)
    sigma_y = np.maximum(sigma_y, 1e-6)
    
    # Temperature-dependent hardening modulus
    H = H0 * (1.0 - temp_hard * T)
    H = np.maximum(H, 1e-6)
    
    # Elastic stress
    sigma_elastic = E * eps
    
    # Plastic strain (beyond yield)
    eps_plastic = np.maximum(eps - eps_y, 0.0)
    
    # Smooth transition: use a sigmoid-like function for the yield point
    # This avoids the kink at the yield point
    transition = 1.0 / (1.0 + np.exp(-(eps - eps_y) / smooth_width))
    transition = np.maximum(transition, 0.0)
    transition = np.minimum(transition, 1.0)
    
    # Plastic stress contribution using combined linear + power-law hardening
    # sigma_plastic = sigma_y + H * eps_plastic + sigma_ref * (eps_plastic / eps_y0)^n_hard
    eps_plastic_norm = eps_plastic / np.maximum(eps_y, 1e-6)
    sigma_plastic = sigma_y + H * eps_plastic + sigma_ref * np.power(eps_plastic_norm, n_hard)
    
    # Blend elastic and plastic responses using the smooth transition
    # For eps < eps_y: mostly elastic
    # For eps > eps_y: mostly plastic
    stress = (1.0 - transition) * sigma_elastic + transition * sigma_plastic
    
    return stress
# EVOLVE-BLOCK-END
