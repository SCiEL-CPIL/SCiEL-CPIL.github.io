"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise-linear model with smooth transitions to capture
    elastic region, yield point, and plastic hardening regime.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent yield strain
    # p4: base yield strain, p5: temperature coefficient
    yield_strain = p4 + p5 * temp
    
    # Temperature-dependent elastic modulus
    # p6: base modulus, p7: temperature coefficient  
    E_temp = p6 + p7 * temp
    
    # Temperature-dependent hardening slope
    # p8: base hardening, p9: temperature coefficient
    H_temp = p8 + p9 * temp
    
    # Elastic region: strain < yield_strain
    # Use smoothstep for smooth transition
    # x = (strain - yield_strain) / width, where width controls transition sharpness
    # We'll use a fixed width of 0.01 for the yield transition
    width = 0.01
    x = (strain - yield_strain) / width
    
    # Smoothstep function: 0 for x<=0, 1 for x>=1, smooth in between
    # This gives us a smooth transition from elastic to plastic
    smooth = np.where(
        x <= 0, 0.0,
        np.where(x >= 1, 1.0, x * x * (3 - 2 * x))
    )
    
    # Elastic stress: E_temp * strain (for strain < yield)
    elastic_stress = E_temp * strain
    
    # Plastic stress: yield_stress + H_temp * (strain - yield_strain) (for strain > yield)
    # yield_stress = E_temp * yield_strain
    yield_stress = E_temp * yield_strain
    plastic_stress = yield_stress + H_temp * (strain - yield_strain)
    
    # Blend between elastic and plastic using smoothstep
    result = (1 - smooth) * elastic_stress + smooth * plastic_stress
    
    # Add temperature offset and quadratic corrections
    # p0: constant offset
    # p1: linear temperature effect
    # p2: quadratic temperature effect
    # p3: temperature-strain interaction
    result = result + p0 + p1 * temp + p2 * temp * temp + p3 * temp * strain
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
