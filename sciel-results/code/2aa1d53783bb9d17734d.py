"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: length 10
    # p0: elastic modulus base
    # p1: plastic hardening coefficient
    # p2: yield stress offset
    # p3: temperature sensitivity (linear)
    # p4: nonlinearity exponent (power law)
    # p5: thermal expansion coefficient
    # p6: interaction term coefficient (strain * temp)
    # p7: saturation limit (denominator offset)
    # p8: cubic strain correction
    # p9: residual stress offset
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability
    # Ramberg-Osgood type form:
    # sigma = E*eps + K*eps^n + temperature effects
    
    # Elastic part: linear in strain with temperature-dependent modulus
    E_eff = p0 * (1.0 + p5 * (temp - 0.5))  # temp normalized around 0.5
    
    # Plastic part: power law with temperature dependence
    # Use a smooth approximation that handles negative strain
    # For negative strain, we want symmetric or antisymmetric behavior
    # Use sign-aware power law
    
    # Base strain for power law - use absolute value with sign preservation
    eps_abs = np.abs(strain)
    eps_sign = np.sign(strain)
    
    # Power law term: K * |eps|^n * sign(eps)
    # n is controlled by p4, ensure n > 0
    n = np.abs(p4) + 0.01  # ensure positive exponent
    K_eff = p1 * (1.0 + p3 * (temp - 0.5))  # temperature-dependent hardening
    
    # Plastic contribution
    plastic = K_eff * np.power(eps_abs + 1e-10, n) * eps_sign
    
    # Yield stress offset with temperature dependence
    yield_offset = p2 * (1.0 + 0.1 * (temp - 0.5))
    
    # Interaction term
    interaction = p6 * strain * temp
    
    # Cubic correction for higher-order effects
    cubic = p8 * strain**3
    
    # Saturation term: prevents unbounded growth
    # sigma_sat = p7 * tanh(sigma_linear / p7)
    linear_part = E_eff * strain + yield_offset + interaction + cubic
    
    # Combine elastic and plastic
    total = linear_part + plastic + p9
    
    # Apply soft saturation to avoid extreme values
    # Use a smooth sigmoid-like saturation
    saturation_scale = np.abs(p7) + 0.1
    total = saturation_scale * np.tanh(total / saturation_scale)
    
    return total
# EVOLVE-BLOCK-END
