"""Symbolic regression with 3 inputs – combined basis with centered polynomials, interactions, and reciprocal terms."""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with 10 features:
    - Centered linear terms (x, y, z)
    - Centered pairwise interactions (xy, xz, yz)
    - Centered three-way interaction (xyz)
    - Centered quadratic terms (x^2, y^2)
    - Reciprocal terms (1/x, 1/y)
    
    Centering around 1.25 (midpoint of [0.5, 2]) improves conditioning for BFGS.
    Reciprocal terms capture rational functional forms.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Center inputs around midpoint of [0.5, 2] = 1.25
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25
    
    # Build 10 features
    f0 = np.ones_like(x)                    # constant
    f1 = xc                                 # centered x
    f2 = yc                                 # centered y
    f3 = zc                                 # centered z
    f4 = xc * yc                            # centered xy interaction
    f5 = xc * zc                            # centered xz interaction
    f6 = yc * zc                            # centered yz interaction
    f7 = xc * yc * zc                       # centered xyz three-way
    f8 = xc * xc                            # centered x^2
    f9 = yc * yc                            # centered y^2
    
    # Add reciprocal terms by replacing two less critical features
    # Actually, let's use a mix: keep the polynomial structure but add reciprocals
    # Let me reconsider: use 1/x and 1/y instead of f8, f9 to capture rational behavior
    
    # Revised: use reciprocal terms for broader coverage
    f8 = 1.0 / x                            # reciprocal x
    f9 = 1.0 / y                            # reciprocal y
    
    pred = (params[0] * f0 + 
            params[1] * f1 + 
            params[2] * f2 + 
            params[3] * f3 + 
            params[4] * f4 + 
            params[5] * f5 + 
            params[6] * f6 + 
            params[7] * f7 + 
            params[8] * f8 + 
            params[9] * f9)
    
    return pred
