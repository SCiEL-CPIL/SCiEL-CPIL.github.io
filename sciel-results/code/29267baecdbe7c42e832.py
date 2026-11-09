"""Combined model: rational structure with tanh saturation for stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines a rational function structure (from current program) with
    a tanh-based hardening term (from inspiration 1) to better capture
    the non-linear behavior of metals.
    
    Model:
    sigma = (p0 + p1*strain + p2*strain^2 + p3*temp + p4*strain*temp)
            / (1 + p5*strain + p6*temp)
            + (p7 + p8*temp) * tanh(p9 * strain + p5 * temp)
    
    Note: Reusing p5 in the tanh argument couples parameters, which may
    reduce flexibility. Let's use distinct parameters for clarity and
    better conditioning.
    
    Revised Model with 10 distinct parameters:
    p0: constant in numerator
    p1: strain coeff in numerator
    p2: strain^2 coeff in numerator
    p3: temp coeff in numerator
    p4: strain*temp coeff in numerator
    p5: strain coeff in denominator
    p6: temp coeff in denominator
    p7: amplitude of tanh term
    p8: temp coeff for tanh amplitude
    p9: slope of tanh argument
    
    sigma = (p0 + p1*strain + p2*strain^2 + p3*temp + p4*strain*temp)
            / (1 + p5*strain + p6*temp)
            + (p7 + p8*temp) * tanh(p9 * strain)
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Rational part
    num = (p0 + p1 * strain + p2 * strain**2 + 
           p3 * temp + p4 * strain * temp)
    
    den = (1.0 + p5 * strain + p6 * temp)
    
    # Safe division
    den_safe = np.where(np.abs(den) < 1e-10, 1e-10 * np.sign(den + 1e-300), den)
    rational_part = num / den_safe
    
    # Tanh hardening part
    # Clip argument to avoid any potential issues, though tanh is stable
    arg = p9 * strain
    hardening = (p7 + p8 * temp) * np.tanh(arg)
    
    stress = rational_part + hardening
    
    return stress
# EVOLVE-BLOCK-END
