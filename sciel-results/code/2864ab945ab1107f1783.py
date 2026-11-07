"""
Symbolic regression with 3 inputs - Robust Quadratic Polynomial Basis.

Strategy:
1. Adopt the 'Linear-in-Parameters' structure from Inspirations 1 & 2.
   This ensures the optimization problem (minimizing MSE) is convex.
   BFGS will reliably converge to the global minimum, avoiding the
   local minima pitfalls of the current program's non-convex Tanh model.
2. Use the Full Second-Order Polynomial Basis.
   This basis spans all monomials of degree <= 2 in x, y, z.
   It is the most general smooth approximation of order 2.
   On the domain [0.5, 2], this provides a very accurate approximation
   for a wide class of smooth functions, including those with
   moderate non-linearity (like the ones hinted at by the Trig score).
3. Ensure numerical stability by flattening params and using direct
   vectorized operations.
"""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a full second-order polynomial basis.
    
    Parameters:
    p[0]: constant term
    p[1]: coefficient for x
    p[2]: coefficient for y
    p[3]: coefficient for z
    p[4]: coefficient for x^2
    p[5]: coefficient for y^2
    p[6]: coefficient for z^2
    p[7]: coefficient for x*y
    p[8]: coefficient for x*z
    p[9]: coefficient for y*z
    """
    p = params.flatten()
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), mode='constant')
    else:
        p = p[:10]
        
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Compute basis terms
    # Using direct operations for speed and clarity
    pred = (c0 + 
            c1 * x + 
            c2 * y + 
            c3 * z + 
            c4 * x * x + 
            c5 * y * y + 
            c6 * z * z + 
            c7 * x * y + 
            c8 * x * z + 
            c9 * y * z)
            
    return pred
