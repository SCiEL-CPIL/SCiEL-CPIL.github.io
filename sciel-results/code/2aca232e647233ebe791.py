"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a generalized elastic-plastic model with smooth transitions:
    - Elastic regime: stress = E * strain (for small strains)
    - Plastic regime: strain hardening via power law
    - Temperature effects: reduce yield strength and modulus
    
    The model combines:
    1. Elastic response with temperature-dependent modulus
    2. Yield point with temperature-dependent yield stress
    3. Strain hardening in the plastic regime
    4. Smooth transition between elastic and plastic regions
    
    params (10 constants):
    p0: reference elastic modulus
    p1: temperature coefficient for modulus
    p2: reference yield stress
    p3: temperature coefficient for yield stress
    p4: strain hardening coefficient
    p5: strain hardening exponent
    p6: transition smoothness parameter
    p7: additional hardening scale
    p8: offset / baseline stress
    p9: temperature offset for normalization
    """
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Ensure p6 is positive for smoothness (use abs to avoid issues)
    p6_safe = np.abs(p6) + 1e-8
    
    # Temperature normalization
    temp_norm = temp - p9
    
    # Temperature factors: reduce stiffness and yield with increasing temp
    # Use bounded exponential to keep things stable
    temp_factor_mod = 1.0 - p1 * temp_norm
    temp_factor_yield = 1.0 - p3 * temp_norm
    
    # Clamp temperature factors to reasonable range to avoid negative modulus/yield
    temp_factor_mod = np.clip(temp_factor_mod, 0.1, 10.0)
    temp_factor_yield = np.clip(temp_factor_yield, 0.1, 10.0)
    
    # Effective elastic modulus
    E_eff = p0 * temp_factor_mod
    
    # Reference yield stress
    yield_stress = p2 * temp_factor_yield
    
    # Yield strain
    yield_strain = np.abs(yield_stress) / (np.abs(E_eff) + 1e-12)
    
    # Absolute strain
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Elastic stress magnitude
    elastic_mag = np.abs(E_eff) * abs_strain
    
    # Smooth minimum between elastic stress and yield stress
    # min(a, b) ≈ -p6 * log(exp(-a/p6) + exp(-b/p6))
    # For numerical stability, use logaddexp
    a = elastic_mag
    b = np.abs(yield_stress)
    
    # Smooth min
    smooth_min_mag = -p6_safe * np.logaddexp(-a / p6_safe, -b / p6_safe)
    
    # Plastic strain (excess beyond yield)
    plastic_strain = np.maximum(0.0, abs_strain - yield_strain)
    
    # Smooth max for plastic strain
    eps = 1e-10
    smooth_plastic = p6_safe * np.logaddexp(plastic_strain / p6_safe, np.zeros_like(plastic_strain))
    
    # Hardening contribution
    # Power-law hardening: p4 * (plastic_strain)^p5
    # Use smooth version
    hardening_mag = p4 * np.power(smooth_plastic + eps, p5) * temp_factor_yield
    
    # Additional hardening term with p7
    hardening_mag2 = p7 * np.power(smooth_plastic + eps, 2.0) * temp_factor_yield
    
    # Total stress magnitude
    total_mag = smooth_min_mag + hardening_mag + hardening_mag2
    
    # Final stress with sign and offset
    stress = sign_strain * total_mag + p8
    
    return stress
# EVOLVE-BLOCK-END
