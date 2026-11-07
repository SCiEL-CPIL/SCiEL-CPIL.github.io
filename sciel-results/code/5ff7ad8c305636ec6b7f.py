"""Symbolic regression using smooth rational function approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Smooth rational function approximation:
    y = (p0 + p1*t + p2*t^2 + p3*t^3 + p4*t^4) / (1 + tanh(q0 + q1*t + q2*t^2 + q3*t^3 + q4*t^4))
    
    where t = x/2 to scale to [-1, 1].
    
    The denominator is bounded in (0, 2] by the tanh function, ensuring it is 
    strictly positive and smooth (differentiable everywhere), which is ideal 
    for BFGS optimization.
    
    params layout (length 10):
      params[0..4] = numerator coefficients p0..p4
      params[5..9] = denominator polynomial coefficients q0..q4
    
    Parameters:
    x: Input array, shape (N,)
    params: Coefficients array, shape (10,)
    
    Returns:
    Predicted y values, shape (N,)
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Scale x to [-1, 1]
    t = x * 0.5
    
    # Extract parameters
    p = params[0:5]
    q = params[5:10]
    
    # Compute numerator: p0 + p1*t + p2*t^2 + p3*t^3 + p4*t^4
    # Using Horner's method for efficiency and stability
    numerator = p[0] + t * (p[1] + t * (p[2] + t * (p[3] + t * p[4])))
    
    # Compute denominator polynomial: q0 + q1*t + q2*t^2 + q3*t^3 + q4*t^4
    denom_poly = q[0] + t * (q[1] + t * (q[2] + t * (q[3] + t * q[4])))
    
    # Apply tanh to bound the denominator
    # tanh output is in (-1, 1), so 1 + tanh is in (0, 2)
    # This ensures the denominator is never zero and is strictly positive
    denominator = 1.0 + np.tanh(denom_poly)
    
    # Compute result
    result = numerator / denominator
    
    # Ensure finite output
    # tanh and polynomials are finite for finite inputs, but just in case of extreme overflow in poly
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
