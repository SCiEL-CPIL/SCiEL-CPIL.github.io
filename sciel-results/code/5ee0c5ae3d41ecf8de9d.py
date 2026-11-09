"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise-linear ramp model with smooth transitions to capture
    elastic region, yield point, and strain hardening regimes.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent scaling for yield point
    # Yield strain decreases with temperature
    yield_strain = p0 * (1.0 - p1 * temp)
    
    # Elastic modulus temperature dependence
    E_temp = p2 * (1.0 - p3 * temp)
    
    # Hardening rate temperature dependence  
    hardening = p4 * (1.0 - p5 * temp)
    
    # Piecewise linear approximation using smooth ramps
    # Ramp 1: Elastic region (linear up to yield)
    ramp1 = np.maximum(strain, 0.0)
    
    # Ramp 2: Yield transition (smooth approximation of max(0, strain - yield_strain))
    # Using softplus for differentiability
    alpha = 200.0  # sharpness of transition
    ramp2 = np.logaddexp(0, alpha * (strain - yield_strain)) / alpha
    
    # Ramp 3: Hardening region (smooth approximation of max(0, strain - 2*yield_strain))
    ramp3 = np.logaddexp(0, alpha * (strain - 2.0 * yield_strain)) / alpha
    
    # Temperature interaction terms
    temp_interaction1 = strain * temp
    temp_interaction2 = ramp2 * temp
    
    # Build the model:
    # p6: base stress offset
    # p7: elastic slope (E_temp * strain)
    # p8: yield plateau coefficient
    # p9: hardening slope
    
    prediction = (
        p6 +
        E_temp * p7 * ramp1 +
        p8 * ramp2 +
        hardening * p9 * ramp3 +
        temp_interaction1 * 0.0 +  # placeholder, not used directly
        temp_interaction2 * 0.0     # placeholder, not used directly
    )
    
    return prediction
# EVOLVE-BLOCK-END
