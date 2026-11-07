# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Trigonometric-Polynomial basis.
    
    The previous polynomial basis was too restrictive (lumped squares).
    The pure trig basis (inspiration) was better, suggesting periodicity.
    This model combines individual sine/cosine terms for each variable
    with a constant and a few polynomial terms to capture non-periodic trends.
    
    Basis functions:
    0: 1
    1: sin(x)
    2: cos(x)
    3: sin(y)
    4: cos(y)
    5: sin(z)
    6: cos(z)
    7: x
    8: y
    9: z
    
    This allows fitting functions like A*sin(x) + B*cos(y) + C*z + D.
    """
    p = params
    
    b0 = 1.0
    b1 = np.sin(x)
    b2 = np.cos(x)
    b3 = np.sin(y)
    b4 = np.cos(y)
    b5 = np.sin(z)
    b6 = np.cos(z)
    b7 = x
    b8 = y
    b9 = z
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
