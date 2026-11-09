import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural ODE style model for nonlinear oscillator acceleration.
    a = v * tanh(p0) - x * (p1 + p2 * tanh(p3 * x^2 + p4 * v))
       + p5 * x^2 * v + p6 * v^2 * x + p7 * x * v^2
       + p8 * sin(p9 * x) * cos(p9 * v)
    """
    p = params
    
    # Ensure parameters are finite
    p = np.where(np.isfinite(p), p, 1.0)
    
    # Core structure: state-dependent damping and stiffness
    # Damping: c(v) = tanh(p0)  (velocity-dependent damping via tanh)
    # Stiffness: k(x, v) = p1 + p2 * tanh(p3 * x^2 + p4 * v)
    
    x2 = x * x
    v2 = v * v
    
    # Damping term
    damp = np.tanh(p[0]) * v
    
    # Stiffness term
    stiff_arg = p[3] * x2 + p[4] * v
    stiff = x * (p[1] + p[2] * np.tanh(stiff_arg))
    
    # Higher-order coupling terms
    cross1 = p[5] * x2 * v
    cross2 = p[6] * v2 * x
    cross3 = p[7] * x * v2
    
    # Oscillatory correction
    phase = p[9] * (x + v)
    osc = p[8] * np.sin(phase) * np.cos(phase)
    
    # Combine: a = -damp - stiff + corrections
    a = -damp - stiff + cross1 + cross2 + cross3 + osc
    
    return a
