"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model for symbolic regression.
    Combines low-degree centered polynomials with trigonometric functions.
    This basis is designed to be well-conditioned for BFGS optimization
    while providing expressive power for both polynomial and periodic trends.
    
    Basis functions:
    1. 1
    2. x
    3. y
    4. x^2 - 1/3 (Centered quadratic)
    5. y^2 - 1/3 (Centered quadratic)
    6. x*y (Interaction)
    7. sin(pi*x)
    8. sin(pi*y)
    9. cos(pi*x) - 0.5 (Centered cosine)
    10. cos(pi*y) - 0.5 (Centered cosine)
    """
    # Pre-compute basis terms
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # 1. Bias
    b0 = np.ones_like(x)
    
    # 2-3. Linear
    b1 = x
    b2 = y
    
    # 4-5. Centered Quadratics
    # E[x^2] on [-1,1] is 1/3. Centering improves conditioning.
    b3 = x**2 - 1.0/3.0
    b4 = y**2 - 1.0/3.0
    
    # 6. Interaction
    b5 = x * y
    
    # 7-8. Sines
    # Using pi * x maps [-1,1] to [-pi, pi], covering a full period of sine/cosine.
    b6 = np.sin(np.pi * x)
    b7 = np.sin(np.pi * y)
    
    # 9-10. Centered Cosines
    # E[cos(pi*x)] is not 0.5 exactly, but 0.5 is a good approximation for centering 
    # to reduce correlation with the bias term.
    b8 = np.cos(np.pi * x) - 0.5
    b9 = np.cos(np.pi * y) - 0.5
    
    # Linear combination
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
