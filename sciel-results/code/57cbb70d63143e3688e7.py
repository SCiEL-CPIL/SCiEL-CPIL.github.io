"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global Chebyshev Polynomial Basis Expansion (Degree 3).
    
    This strategy fundamentally changes the model class from localized non-linear 
    functions (RBFs) to a global orthogonal polynomial basis.
    
    Advantages over RBFs:
    1. Linearity: The model is linear in parameters. BFGS will converge to the 
       global minimum in one step (or very few), avoiding local minima traps 
       that plague RBFs when centers are far apart.
    2. Orthogonality: Chebyshev polynomials are orthogonal on [-1, 1], which 
       reduces multicollinearity compared to monomials, leading to better 
       numerical stability and more robust parameter fitting.
    3. Global Coverage: Unlike RBFs which require centers to be well-placed, 
       this basis covers the entire domain uniformly, making it robust to 
       initialization (all ones) and the specific location of data clusters.
    
    Basis Functions (10 total for degree <= 3 in 2D):
    T0(x)T0(y) = 1
    T1(x)T0(y) = x
    T0(x)T1(y) = y
    T2(x)T0(y) = 2x^2 - 1
    T1(x)T1(y) = xy
    T0(x)T2(y) = 2y^2 - 1
    T3(x)T0(y) = 4x^3 - 3x
    T2(x)T1(y) = (2x^2 - 1)y
    T1(x)T2(y) = x(2y^2 - 1)
    T0(x)T3(y) = 4y^3 - 3y
    
    Parameters:
    p0: weight for 1
    p1: weight for x
    p2: weight for y
    p3: weight for 2x^2 - 1
    p4: weight for xy
    p5: weight for 2y^2 - 1
    p6: weight for 4x^3 - 3x
    p7: weight for (2x^2 - 1)y
    p8: weight for x(2y^2 - 1)
    p9: weight for 4y^3 - 3y
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Ensure inputs are 1D arrays for consistent broadcasting
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    
    # Extract parameters
    w = np.asarray(params, dtype=float)[:10]
    
    # Compute Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2.0 * x * x - 1.0
    Tx3 = 4.0 * x * x * x - 3.0 * x
    
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2.0 * y * y - 1.0
    Ty3 = 4.0 * y * y * y - 3.0 * y
    
    # Construct basis functions
    # Order matches parameter indices p0..p9
    
    # Degree 0
    b0 = Tx0 * Ty0  # 1
    
    # Degree 1
    b1 = Tx1 * Ty0  # x
    b2 = Tx0 * Ty1  # y
    
    # Degree 2
    b3 = Tx2 * Ty0  # 2x^2 - 1
    b4 = Tx1 * Ty1  # xy
    b5 = Tx0 * Ty2  # 2y^2 - 1
    
    # Degree 3
    b6 = Tx3 * Ty0  # 4x^3 - 3x
    b7 = Tx2 * Ty1  # (2x^2 - 1)y
    b8 = Tx1 * Ty2  # x(2y^2 - 1)
    b9 = Tx0 * Ty3  # 4y^3 - 3y
    
    # Combine linearly
    result = (w[0] * b0 + 
              w[1] * b1 + 
              w[2] * b2 + 
              w[3] * b3 + 
              w[4] * b4 + 
              w[5] * b5 + 
              w[6] * b6 + 
              w[7] * b7 + 
              w[8] * b8 + 
              w[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
