import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed Basis Linear Regression.
    
    The current program uses a small MLP which is non-convex and hard for BFGS to optimize 
    from a single start point (all ones). This strategy uses a fixed set of 10 basis 
    functions that span a rich function space (polynomial up to degree 2 + some higher 
    order/interaction terms). 
    
    The model is: f(x, y, z) = sum_{i=0}^{9} params[i] * phi_i(x, y, z)
    
    Since the relationship between params and the output is linear, the optimization 
    landscape is convex (a paraboloid). BFGS will converge quickly and reliably to the 
    optimal coefficients for this fixed basis. This avoids the local minima traps of the 
    MLP while providing more expressive power than a simple quadratic if the basis is 
    chosen well.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: x
    2: y
    3: z
    4: x^2
    5: y^2
    6: z^2
    7: xy
    8: xz
    9: yz
    
    This covers all quadratic terms. If the target is more complex, we might need more 
    terms, but we are limited to 10 params. A standard quadratic in 3 vars has 10 coeffs.
    This is the maximum capacity for a pure quadratic model.
    
    To potentially capture more non-linearity without increasing param count, we could 
    use a different basis, e.g., including some cubic terms or normalized polynomials, 
    but the quadratic basis is the most standard and well-conditioned starting point 
    for [0.5, 2] inputs.
    """
    try:
        n = len(x)
        
        # Construct the design matrix Phi of shape (n, 10)
        # Columns correspond to the basis functions
        
        # 0: 1
        col0 = np.ones(n)
        # 1: x
        col1 = x
        # 2: y
        col2 = y
        # 3: z
        col3 = z
        # 4: x^2
        col4 = x * x
        # 5: y^2
        col5 = y * y
        # 6: z^2
        col6 = z * z
        # 7: xy
        col7 = x * y
        # 8: xz
        col8 = x * z
        # 9: yz
        col9 = y * z
        
        Phi = np.column_stack((col0, col1, col2, col3, col4, col5, col6, col7, col8, col9))
        
        # Output is linear combination: Phi @ params
        output = Phi @ params
        
        # Safety check
        if not np.all(np.isfinite(output)):
            return np.zeros(n)
            
        return output

    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
