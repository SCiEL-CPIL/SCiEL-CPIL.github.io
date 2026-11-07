import numpy as np
import time

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalization
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 20.0) / 20.0
    ph_n = (pH - 2.0) / 10.0
    
    # Substrate: Monod-like
    # Use exp for positive parameters
    ks = np.exp(p[1]) + 1e-6
    hill = p[9] # Revert to free hill, as parent did
    
    # Safe power
    # s_n is in [0.2, 1]. hill can be any real.
    # If hill is very large positive, s_n**hill -> 0.
    # If hill is very large negative, s_n**hill -> inf.
    # To be safe, clip hill?
    hill = np.clip(hill, -2.0, 5.0)
    
    monod = s_n**hill / (ks**hill + s_n**hill)
    
    # Temperature: Gaussian
    t_opt = p[2]
    t_width = np.exp(p[3]) + 1e-6
    temp_factor = np.exp(-0.5 * ((t_n - t_opt) / t_width)**2)
    
    # pH: Gaussian
    ph_opt = p[4]
    ph_width = np.exp(p[5]) + 1e-6
    ph_factor = np.exp(-0.5 * ((ph_n - ph_opt) / ph_width)**2)
    
    # Density: Logistic inhibition
    dens_term = 1.0 / (1.0 + np.exp(p[6] * b_n - p[7]))
    
    # Combine
    result = p[0] * monod * temp_factor * ph_factor * dens_term + p[8]
    
    return result
