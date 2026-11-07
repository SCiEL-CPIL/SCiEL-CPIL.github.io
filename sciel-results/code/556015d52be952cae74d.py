"""
Strategy: Use a full cubic polynomial basis in x and y.
This utilizes all 10 parameters to represent any function of the form:
f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 + p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3

This basis is linear in the parameters, which makes the optimization landscape convex (or at least very well-behaved for BFGS) when minimizing MSE.
Since the domain is [-1, 1], cubic polynomials provide a good approximation for many smooth functions.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Construct the prediction
    # Basis: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    pred = (p0 + 
            p1 * x + 
            p2 * y + 
            p3 * x2 + 
            p4 * xy + 
            p5 * y2 + 
            p6 * x3 + 
            p7 * x2y + 
            p8 * xy2 + 
            p9 * y3)
            
    return pred
# EVOLVE-BLOCK-END
