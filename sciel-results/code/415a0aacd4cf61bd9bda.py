import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a smooth elastic-plastic model with temperature dependence.
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    E0 = params[0]
    cE = params[1]
    H0 = params[2]
    cH = params[3]
    eps_y0 = params[4]
    ceps_y = params[5]
    k = params[6]
    b0 = params[7]
    cb = params[8]
    # params[9] is reserved, let's use it to scale k slightly with T or as a pure constant
    # Let's use it to allow a small offset in the yield strain itself or just add to bias
    # Let's add it to the bias term to have two bias terms if needed, or use it for a 
    # quadratic strain correction term: params[9] * eps**2
    # A quadratic term can help if the hardening is not perfectly linear.
    
    # Temperature dependent properties
    E = E0 * (1.0 + cE * T)
    H = H0 * (1.0 + cH * T)
    eps_y = eps_y0 * (1.0 + ceps_y * T)
    bias = b0 + cb * T + params[9]
    
    # Smooth transition term
    # S = (1 + tanh(k * (eps - eps_y))) / 2
    # This goes from 0 to 1.
    # We want to subtract (E-H) * S * (eps - eps_y) from E*eps
    # To avoid numerical issues with tanh for very large arguments, np.tanh is safe.
    
    diff = eps - eps_y
    # Ensure k is positive for stable convergence? BFGS might find negative k.
    # tanh(-x) = -tanh(x). If k is negative, the transition is reversed.
    # Physically, we want the transition to go from Elastic to Plastic as strain increases.
    # So k should be positive. However, the optimizer might struggle if the landscape is flat.
    # We can use abs(k) to ensure the shape is correct, but this introduces a non-differentiability at 0.
    # Better to let it be, or initialize close to positive.
    # The prompt says params start from all ones. So k=1 initially.
    
    s = 0.5 * (1.0 + np.tanh(k * diff))
    
    # Elastic stress
    sigma_elastic = E * eps
    
    # Plastic correction
    # In elastic region (s~0), correction is 0.
    # In plastic region (s~1), correction is (E-H)*(eps - eps_y).
    # Total stress = E*eps - (E-H)*(eps - eps_y) = E*eps - E*eps + E*eps_y + H*eps - H*eps_y = H*eps + (E-H)*eps_y
    # This matches the bilinear model.
    
    correction = (E - H) * s * diff
    
    sigma = sigma_elastic - correction + bias
    
    return sigma
