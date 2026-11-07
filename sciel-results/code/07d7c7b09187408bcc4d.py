import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for consistent operations, though it should be
    # Ensure params is a 1D array
    p = np.asarray(params).flatten()
    
    # Basis functions
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Trig terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Sigmoid/Saturation
    tanh_x = np.tanh(x)
    
    # Gaussian bump
    exp_neg_x2 = np.exp(-0.5 * x2)
    
    # Linear combination
    # p[0] to p[5]: polynomial
    # p[6], p[7]: trig
    # p[8]: tanh
    # p[9]: gaussian
    
    result = (p[0] + 
              p[1] * x + 
              p[2] * x2 + 
              p[3] * x3 + 
              p[4] * x4 + 
              p[5] * x5 + 
              p[6] * sin_x + 
              p[7] * cos_x + 
              p[8] * tanh_x + 
              p[9] * exp_neg_x2)
              
    return result
