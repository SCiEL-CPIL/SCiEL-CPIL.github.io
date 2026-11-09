"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a rich basis of 10 features:
    - Chebyshev-like scaled polynomials (u to u^4)
    - Exponentials (exp(u), exp(-u))
    - Trigonometric functions (sin/cos with different frequencies)
    
    x is scaled to u in [-1, 1] to improve numerical stability for BFGS.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    # u = (x - 3.05) / 2.95
    u = (x - 3.05) / 2.95
    
    # Construct basis terms
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-5. Polynomials in u
    b1 = u
    b2 = u * u
    b3 = b2 * u
    b4 = b3 * u
    
    # 6-7. Exponentials
    # Using u directly keeps values bounded between exp(-1) and exp(1)
    b5 = np.exp(u)
    b6 = np.exp(-u)
    
    # 8-10. Trigonometric
    # Frequencies chosen to cover potential oscillations in the domain
    # 2*pi corresponds to one period across the range
    b7 = np.sin(2.0 * np.pi * u)
    b8 = np.cos(2.0 * np.pi * u)
    b9 = np.sin(4.0 * np.pi * u)
    
    # Combine with parameters
    # params is length 10
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    y = (p0 * b0 + 
         p1 * b1 + 
         p2 * b2 + 
         p3 * b3 + 
         p4 * b4 + 
         p5 * b5 + 
         p6 * b6 + 
         p7 * b7 + 
         p8 * b8 + 
         p9 * b9)
         
    return y
# EVOLVE-BLOCK-END
