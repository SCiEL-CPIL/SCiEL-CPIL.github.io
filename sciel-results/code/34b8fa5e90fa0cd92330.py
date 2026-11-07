"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Center x to improve conditioning of polynomial terms
    # Range is [0.1, 6], center is 3.05
    u = x - 3.05
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear (centered)
    b1 = u
    # 3. Quadratic (centered)
    b2 = u ** 2
    # 4. Cubic (centered)
    b3 = u ** 3
    
    # 5. Exponential growth
    b4 = np.exp(x)
    # 6. Exponential decay
    b5 = np.exp(-x)
    
    # 7. Sine
    b6 = np.sin(x)
    # 8. Cosine
    b7 = np.cos(x)
    
    # 9. Sine 2x
    b8 = np.sin(2 * x)
    # 10. Cosine 2x
    b9 = np.cos(2 * x)
    
    # Linear combination
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
# EVOLVE-BLOCK-END
