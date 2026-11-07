"""Symbolic regression with 2 inputs using a rich basis and robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model with the given parameters.
    params: array of length 10 (as required by the evaluator).
    We use a basis of 10 features and map the 10 params to them.
    """
    # Build the 10-feature basis
    # We choose a diverse set of features that can represent many smooth functions.
    # Note: The evaluator passes params of length 10. We use all 10.
    
    # Feature design:
    # 0: 1 (bias)
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: x*y
    # 6: sin(x)
    # 7: sin(y)
    # 8: cos(x)
    # 9: cos(y)
    
    # However, the seed used params[0]*x + params[1]*y + params[2].
    # We can define our own basis. The evaluator just passes 10 params.
    # We need to make sure the basis is linear in params.
    
    p = params
    
    # Let's use a basis that includes polynomials up to degree 3 and some trig.
    # But we only have 10 params.
    # Let's try:
    # 1, x, y, x^2, y^2, x*y, x^3, y^3, x^2*y, x*y^2
    # This is a complete polynomial basis of degree 3 (10 terms).
    
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x**2
    f5 = y**2
    f6 = x * y
    f7 = x**3
    f8 = y**3
    f9 = x**2 * y
    f10 = x * y**2
    
    return (p[0] * f1 + p[1] * f2 + p[2] * f3 + p[3] * f4 + 
            p[4] * f5 + p[5] * f6 + p[6] * f7 + p[7] * f8 + 
            p[8] * f9 + p[9] * f10)

# EVOLVE-BLOCK-END
