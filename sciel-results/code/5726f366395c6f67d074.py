import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed nonlinear oscillator model.
    
    Structure:
    a = p4 * x * (1 + p0 * x^2) + p5 * v * (1 + p1 * x^2) + p2 * x * v + p6 * v^2 + p3
    + p7 * x^5 + p8 * x^2 * v^2 + p9 * v^4
    
    This captures:
    - Linear and cubic restoring force (Duffing)
    - Linear and amplitude-dependent damping (Van der Pol)
    - Cross-coupling and higher-order drag terms
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v4 = v2 * v2
    
    # Restoring force: Linear + Cubic (Duffing)
    # x * (1 + p0 * x^2) = x + p0 * x^3
    restoring = p4 * x * (1.0 + p0 * x2)
    
    # Damping: Linear + Amplitude-dependent (Van der Pol)
    # v * (1 + p1 * x^2) = v + p1 * x^2 * v
    damping = p5 * v * (1.0 + p1 * x2)
    
    # Cross terms and higher orders
    cross = p2 * x * v
    drag_v2 = p6 * v2
    higher_rest = p7 * x5
    higher_damp = p8 * x2 * v2
    higher_drag = p9 * v4
    
    return restoring + damping + cross + drag_v2 + higher_rest + higher_damp + higher_drag + p3
