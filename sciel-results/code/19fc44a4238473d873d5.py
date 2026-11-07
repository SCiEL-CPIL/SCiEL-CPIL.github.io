import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalization
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 20.0) / 20.0
    ph_n = (pH - 2.0) / 10.0
    
    # Substrate: Monod-like but with a Hill coefficient that is safe
    # Use a softplus-like or just squared for Ks to ensure positivity
    # Hill coefficient: use abs to ensure positive, plus small constant
    # To avoid overflow in s_n**hill, clip hill or use a safer power
    # Let's use a standard Monod (hill=1) but with a flexible Ks
    # Or use a bounded hill: hill = 1.0 + 0.5 * np.tanh(p[9])
    # This keeps hill in [0.5, 1.5], which is safe for power
    
    ks = p[1] * p[1] + 1e-6
    hill = 1.0 + 0.5 * np.tanh(p[9])
    
    # Safe power: s_n is in [0.2, 1] (since s in [1,5], s_n in [0.2,1])
    # So s_n**hill is safe. ks**hill is also safe.
    monod = s_n**hill / (ks**hill + s_n**hill)
    
    # Temperature: Gaussian
    t_opt = p[2]
    t_width = p[3] * p[3] + 1e-6
    temp_factor = np.exp(-0.5 * ((t_n - t_opt) / t_width)**2)
    
    # pH: Gaussian
    ph_opt = p[4]
    ph_width = p[5] * p[5] + 1e-6
    ph_factor = np.exp(-0.5 * ((ph_n - ph_opt) / ph_width)**2)
    
    # Density: Logistic inhibition
    # density_term = 1.0 - 1.0/(1.0 + exp(p[6]*(b_n - p[7])))
    # Or simpler: 1.0 / (1.0 + exp(p[6]*b_n - p[7]))
    # Let's use a logistic function for density inhibition
    # p[6] is slope, p[7] is midpoint
    dens_term = 1.0 / (1.0 + np.exp(p[6] * b_n - p[7]))
    
    # Combine
    # p[0] is global scale
    # p[8] is an additive offset? Or keep it multiplicative?
    # Let's add p[8] as an additive term to handle baseline
    # result = p[0] * monod * temp_factor * ph_factor * dens_term + p[8]
    
    # But if we add p[8], it might help with the lower bound.
    # However, the target range is [-0.00277, 1.553].
    # Let's try: result = p[0] * monod * temp_factor * ph_factor * dens_term
    # and see if p[8] can be used for something else or just kept for flexibility.
    
    # Let's use p[8] as a second additive term or just keep the current structure
    # and use p[8] in the density term?
    
    # Let's try a slightly different density term:
    # dens_term = p[8] / (p[8] + b_n**p[6])
    # This is a Michaelis-Menten like term for density.
    # p[8] is Km-like, p[6] is Hill-like.
    
    # Let's stick to the logistic for now, and use p[8] as an offset.
    result = p[0] * monod * temp_factor * ph_factor * dens_term + p[8]
    
    return result
