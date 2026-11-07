import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters polynomial model for stress-strain.
    
    This model is linear in the parameters, which makes the MSE loss function
    convex. This allows scipy.optimize.minimize with BFGS to find the global
    minimum reliably and quickly, avoiding the local minima traps of non-linear
    models like the tanh-switch bilinear model.
    
    Basis functions are scaled to improve numerical conditioning.
    """
    s = strain
    t = temp
    p = params
    
    # Normalize inputs to [0, 1] range based on training data bounds
    # s_min = -0.001911, s_max = 0.9636
    # t_min = 0.06667, t_max = 1.0
    
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_norm = (s - s_min) / (s_max - s_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Clip to [0, 1] to prevent extrapolation issues during optimization
    # if BFGS steps outside the training range, though it should stay close.
    # Actually, for a linear model, extrapolation is fine mathematically, 
    # but clipping ensures stability if the optimizer wanders.
    s_norm = np.clip(s_norm, 0.0, 1.0)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    
    # Construct basis functions
    # 1. Intercept
    phi0 = np.ones_like(s_norm)
    
    # 2. Linear Strain
    phi1 = s_norm
    
    # 3. Linear Temp
    phi2 = t_norm
    
    # 4. Interaction S*T
    phi3 = s_norm * t_norm
    
    # 5. Quadratic Strain
    phi4 = s_norm * s_norm
    
    # 6. Quadratic Strain * Temp
    phi5 = phi4 * t_norm
    
    # 7. Cubic Strain
    phi6 = phi4 * s_norm
    
    # 8. Cubic Strain * Temp
    phi7 = phi6 * t_norm
    
    # 9. Quartic Strain
    phi8 = phi6 * s_norm
    
    # 10. Quartic Strain * Temp
    phi9 = phi8 * t_norm
    
    # Linear combination
    result = (p[0] * phi0 + 
              p[1] * phi1 + 
              p[2] * phi2 + 
              p[3] * phi3 + 
              p[4] * phi4 + 
              p[5] * phi5 + 
              p[6] * phi6 + 
              p[7] * phi7 + 
              p[8] * phi8 + 
              p[9] * phi9)
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
