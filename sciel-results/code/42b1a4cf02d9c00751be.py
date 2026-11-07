import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-constrained rational function.
    The acceleration is modeled as a ratio of two low-degree polynomials to capture
    saturation and softening nonlinearities.
    
    Inputs:
    t: time
    x: position
    v: velocity
    params: array of 10 fitted constants
    
    Returns:
    y: acceleration
    """
    # Scale inputs to O(1) to improve conditioning for BFGS
    # Based on training ranges: x ~ [-0.21, 0.17], v ~ [-0.34, 0.36]
    x_scale = 0.2
    v_scale = 0.3
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Trigonometric functions of time
    st = np.sin(t)
    ct = np.cos(t)
    
    # Unpack parameters: 6 for numerator, 4 for denominator
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Numerator terms (N(x,v,t))
    # Linear and quadratic state terms plus forcing
    n0 = xs
    n1 = xs * xs
    n2 = vs
    n3 = st
    n4 = xs * vs
    n5 = vs * vs
    
    numerator = (p0 * n0 + 
                 p1 * n1 + 
                 p2 * n2 + 
                 p3 * n3 + 
                 p4 * n4 + 
                 p5 * n5)
    
    # Denominator terms (D(x,v,t))
    # Constant plus quadratic terms to create saturation/softening
    d0 = 1.0
    d1 = xs * xs
    d2 = vs * vs
    d3 = xs * vs
    
    denominator = (p6 * d0 + 
                   p7 * d1 + 
                   p8 * d2 + 
                   p9 * d3)
    
    # Prevent division by zero by adding a small epsilon
    eps = 1e-8
    denom_safe = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + 1e-16), denominator)
    # Ensure denominator doesn't flip sign unexpectedly near zero
    denom_safe = np.where(denominator == 0, eps, denom_safe)
    
    result = numerator / denom_safe
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
        
    return result
