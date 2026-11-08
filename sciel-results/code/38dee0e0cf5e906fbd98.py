import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration for a nonlinear oscillator combining smooth bounded
    nonlinearities (tanh) with physical polynomial terms (Duffing, Van der Pol,
    cubic damping).
    
    Structure:
    - Linear terms: p1*x + p2*v
    - Bounded stiffness: p3*tanh(p0*x)
    - Duffing cubic: p4*x^3
    - Bounded damping: p5*tanh(p6*v)
    - Cubic damping: p7*v^3
    - Van der Pol coupling: p8*x^2*v
    - Higher order cross term: p9*x*v^2
    
    The tanh terms provide smooth, bounded nonlinearities that help BFGS
    converge, while the polynomial terms capture the standard nonlinear
    oscillator physics.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    
    # Smooth bounded nonlinearities
    tanh_x = np.tanh(p0 * x)
    tanh_v = np.tanh(p6 * v)
    
    # Construct the model:
    # Linear stiffness and damping
    # Bounded nonlinear stiffness and damping via tanh
    # Cubic Duffing stiffness
    # Cubic damping
    # Van der Pol coupling (x^2 * v)
    # Higher order cross coupling (x * v^2)
    
    return (p1 * x + 
            p2 * v + 
            p3 * tanh_x + 
            p4 * x3 + 
            p5 * tanh_v + 
            p7 * v2 * v + 
            p8 * x2 * v + 
            p9 * x * v2)
