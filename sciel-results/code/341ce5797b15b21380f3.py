import numpy as np
import time

def equation(b, s, temp, pH, params):
    p = params
    
    # Deadline guard
    start_time = time.time()
    
    # Normalization
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 20.0) / 20.0
    ph_n = (pH - 2.0) / 10.0
    
    # Substrate: Monod-like
    ks = np.exp(p[1]) + 1e-6
    hill = np.clip(p[9], -2.0, 5.0)
    
    # Safe power
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
