import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physically-motivated basis
    with normalized coordinates for better conditioning.
    
    The basis includes:
    - Linear restoring force (x)
    - Damping (v)
    - Nonlinear stiffness (x^3, x^5)
    - Nonlinear damping (v^3, x*v^2)
    - Cross terms (x*v, x^2*v, x^2)
    - Constant
    """
    p = params
    
    # Normalize to O(1) scale
    # x range: [-0.5728, 0.5275] -> scale by ~0.55
    # v range: [-0.1754, 0.2821] -> scale by ~0.28
    x_s = x / 0.55
    v_s = v / 0.28
    
    # Compute powers
    x_s2 = x_s * x_s
    x_s3 = x_s2 * x_s
    x_s5 = x_s3 * x_s2
    v_s2 = v_s * v_s
    v_s3 = v_s2 * v_s
    x_v = x_s * v_s
    x2_v = x_s2 * v_s
    x_v2 = x_s * v_s2
    x2 = x_s2
    
    # Basis functions - physically motivated for oscillator
    # f0: linear restoring force
    f0 = x_s
    # f1: linear damping
    f1 = v_s
    # f2: cubic stiffness (Duffing-like)
    f2 = x_s3
    # f3: quintic stiffness (hardening/softening)
    f3 = x_s5
    # f4: cubic damping
    f4 = v_s3
    # f5: velocity-squared damping (quadratic drag)
    f5 = x_v2
    # f6: cross term
    f6 = x_v
    # f7: position-squared damping
    f7 = x2_v
    # f8: quadratic position term
    f8 = x2
    # f9: constant offset
    f9 = 1.0
    
    return (p[0] * f0 + p[1] * f1 + p[2] * f2 + p[3] * f3 + 
            p[4] * f4 + p[5] * f5 + p[6] * f6 + p[7] * f7 + 
            p[8] * f8 + p[9] * f9)
