import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using tanh-based nonlinearities.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basic terms
    term_x = x
    term_v = v
    
    # Nonlinear spring (tanh saturation)
    term_spring_nonlin = np.tanh(p0 * x)
    
    # Nonlinear damping (tanh saturation)
    term_damp_nonlin = np.tanh(p1 * v)
    
    # Cubic-like spring: x * tanh(p * x^2) ~ p * x^3 for small x
    term_spring_cubic = x * np.tanh(p2 * x * x)
    
    # Cubic-like damping: v * tanh(p * v^2) ~ p * v^3 for small x
    term_damp_cubic = v * np.tanh(p3 * v * v)
    
    # Cross terms
    term_cross_lin = x * v
    term_cross_nonlin = np.tanh(p4 * x * v)
    
    # Quadratic terms (allowing for asymmetry)
    term_x_quad = x * x
    term_v_quad = v * v
    
    # Combine
    # p5, p6, p7, p8, p9 are weights for the above? 
    # I have 10 params. 
    # Let's assign:
    # p0: weight for term_spring_nonlin? No, p0 is inside tanh.
    # I need weights for the basis functions too.
    # I only have 10 params total.
    # If I use p0..p4 inside tanh, I have 5 left for weights.
    # But I have 8 basis terms above.
    # I need to fix the structure to use exactly 10 params.
    
    # Strategy: Use 5 params for scaling inside tanh, 5 for weights?
    # Or mix them.
    
    # Let's try:
    # y = w0 * tanh(s0 * x) + w1 * tanh(s1 * v) + w2 * x * tanh(s2 * x**2) + w3 * v * tanh(s3 * v**2) + w4 * tanh(s4 * x * v)
    # This uses 10 params: w0..w4, s0..s4.
    # But this misses linear terms x, v explicitly. 
    # tanh(s*x) approximates s*x. So w0*tanh(s0*x) approximates w0*s0*x.
    # This effectively creates a linear term.
    # The quadratic terms x^2, v^2 are missing. 
    # If the system is symmetric, x^2, v^2 should be 0.
    # Let's assume symmetry first.
    
    return (p0 * np.tanh(p1 * x) + 
            p2 * np.tanh(p3 * v) + 
            p4 * x * np.tanh(p5 * x * x) + 
            p6 * v * np.tanh(p7 * v * v) + 
            p8 * np.tanh(p9 * x * v))
