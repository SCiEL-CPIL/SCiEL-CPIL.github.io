import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with a physics-informed basis.
    Includes linear, cubic, and sinusoidal restoring forces,
    linear, cubic, and saturated damping, and interaction terms 
    like Van der Pol (x^2*v) and cross-coupling (x*v, x*v^2).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Position terms (Restoring Force)
    x_term = x
    x_cubed = x * x * x
    x_sin = np.sin(x)
    
    # Velocity terms (Damping)
    v_term = v
    v_cubed = v * v * v
    v_tanh = np.tanh(v)
    
    # Interaction terms
    x_v = x * v
    x_sq_v = x * x * v  # Van der Pol style
    x_v_sq = x * v * v  # Cross coupling
    
    # Construct the equation
    # p0: x
    # p1: x^3
    # p2: sin(x)
    # p3: v
    # p4: v^3
    # p5: tanh(v)
    # p6: x^2*v
    # p7: x*v
    # p8: x*v^2
    # p9: constant
    
    return p0 * x_term + p1 * x_cubed + p2 * x_sin + p3 * v_term + p4 * v_cubed + p5 * v_tanh + p6 * x_sq_v + p7 * x_v + p8 * x_v_sq + p9
