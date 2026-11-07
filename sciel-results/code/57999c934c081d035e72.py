import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Structural change: Generalized additive model with non-linear transforms
    # instead of multiplicative interaction model
    
    # Substrate: Use both linear and saturating terms
    ks = np.exp(p[0])  # Half-saturation constant
    s_linear = p[1] * s
    s_sat = p[2] * s / (ks + s)  # Monod-like saturation
    
    # Temperature: Use polynomial instead of Gaussian (more flexible)
    t_center = p[3]
    t_scale = np.exp(p[4])  # Positive scale
    t_normalized = (temp - t_center) / t_scale
    temp_poly = 1.0 - 0.5 * t_normalized**2 + p[5] * t_normalized**4  # Quartic with negative quadratic
    
    # pH: Also use polynomial
    ph_center = p[6]
    ph_scale = np.exp(p[7])  # Positive scale
    ph_normalized = (pH - ph_center) / ph_scale
    ph_poly = 1.0 - 0.5 * ph_normalized**2 + p[8] * ph_normalized**4
    
    # Biomass/buffer: linear term
    b_term = p[9] * b
    
    # Combine all terms additively
    # Note: This uses all 10 parameters
    # Parameters 0-2: substrate
    # Parameters 3-5: temperature  
    # Parameters 6-8: pH
    # Parameter 9: biomass
    result = s_linear + s_sat * temp_poly * ph_poly + b_term
    
    return result
