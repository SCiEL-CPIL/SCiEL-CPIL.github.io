import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 learnable parameters.
    Uses a basis of linear, inverse, and ratio terms to capture rational function structures.
    """
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-4. Linear terms
    b1 = x
    b2 = y
    b3 = z
    
    # 5-7. Inverse terms
    b4 = 1.0 / x
    b5 = 1.0 / y
    b6 = 1.0 / z
    
    # 8-10. Ratio terms (forward)
    b7 = x / y
    b8 = x / z
    b9 = y / z
    
    # Combine with learnable parameters
    # params is an array of length 10
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
            params[4] * b4 + params[5] * b5 + params[6] * b6 + params[7] * b7 +
            params[8] * b8 + params[9] * b9)
