"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for polynomial conditioning
    u = x / 3.0
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = u
    
    # 3. Quadratic (Chebyshev T2)
    b2 = 2 * u**2 - 1
    
    # 4. Cubic (Chebyshev T3)
    b3 = 4 * u**3 - 3 * u
    
    # 5. Quartic (Chebyshev T4)
    b4 = 8 * u**4 - 8 * u**2 + 1
    
    # 6. Sine (standard frequency)
    b5 = np.sin(x)
    
    # 7. Cosine (standard frequency)
    b6 = np.cos(x)
    
    # 7. Gaussian
    b7 = np.exp(-x**2)
    
    # 9. Tanh
    b8 = np.tanh(x)
    
    # 10. Interaction term
    b9 = x * np.sin(x)
    
    # Combine
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
