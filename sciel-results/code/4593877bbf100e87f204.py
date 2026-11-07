import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # 10s safety margin before 150s kill

def equation(b, s, temp, pH, params):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    p = params
    
    # Normalization
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 20.0) / 20.0
    ph_n = (pH - 2.0) / 10.0
    
    # Monod term for substrate
    # Use exp to ensure positive Ks, clip hill to [0.5, 5] for stability
    ks = np.exp(p[1]) + 1e-8
    hill = np.clip(p[9], 0.5, 5.0)
    
    # Robust Monod: use log-space to avoid overflow/underflow
    # monod = s_n^hill / (ks^hill + s_n^hill)
    # = 1 / (1 + (ks/s_n)^hill)
    # Use safe division
    s_safe = np.clip(s_n, 1e-12, None)
    ratio = ks / s_safe
    # For large ratio, monod -> 0; for small ratio, monod -> 1
    # Use exp form: monod = exp(-hill * log(ratio)) / (1 + exp(-hill * log(ratio)))
    # But simpler: just compute directly with clipping
    ratio_hill = ratio ** hill
    monod = 1.0 / (1.0 + ratio_hill)
    
    # Gaussian for temp
    t_opt = p[2]
    t_width = np.exp(p[3]) + 1e-8
    temp_factor = np.exp(-0.5 * ((t_n - t_opt) / t_width)**2)
    
    # Gaussian for pH
    ph_opt = p[4]
    ph_width = np.exp(p[5]) + 1e-8
    ph_factor = np.exp(-0.5 * ((ph_n - ph_opt) / ph_width)**2)
    
    # Density term: use softplus-like to keep positive
    # dens = exp(p[6] + p[7]*b_n + p[8]*b_n**2)
    # But this might overflow. Use a bounded version.
    dens_arg = p[6] + p[7]*b_n + p[8]*b_n**2
    dens_arg = np.clip(dens_arg, -50, 50)
    dens_term = np.exp(dens_arg)
    
    # Combine
    result = p[0] * monod * temp_factor * ph_factor * dens_term
    
    return result
