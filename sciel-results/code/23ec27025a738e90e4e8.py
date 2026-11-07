import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametrically excited Duffing oscillator model.
    
    The acceleration is modeled as a function of position x, velocity v, and time t.
    We use a basis of physics-inspired terms:
    - Nonlinear restoring force: x, x^3
    - Damping terms: v, x^2*v (Van der Pol style), x*v
    - Parametric excitation: sin(t)*x, cos(t)*v, sin(t), cos(t)
    - Cross terms: sin(t)*x^2, cos(t)*v^2, sin(t)*v, cos(t)*x
    
    Inputs are normalized to improve conditioning.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalization
    # x range approx [-0.2056, 0.1749] -> center -0.015, scale 0.19
    # v range approx [-0.3391, 0.3624] -> center 0.011, scale 0.35
    xn = (x + 0.015) / 0.19
    vn = (v - 0.011) / 0.35

    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn

    # Time-dependent terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    # Basis functions for parametrically excited Duffing oscillator:
    # Core nonlinear dynamics:
    b0 = 1.0                     # Offset
    b1 = xn                      # Linear restoring
    b2 = xn3                     # Cubic restoring (Duffing)
    b3 = vn                      # Linear damping
    b4 = xn2 * vn                # Van der Pol damping
    
    # Parametric excitation terms (time-dependent modulation):
    b5 = sin_t * xn              # Parametric stiffness modulation
    b6 = cos_t * vn              # Parametric damping modulation
    b7 = sin_t * vn              # Time-dependent forcing on velocity
    b8 = cos_t * xn              # Time-dependent forcing on position
    b9 = sin_t * cos_t           # Cross time term (captures phase interactions)

    return (p0 * b0 + 
            p1 * b1 + 
            p2 * b2 + 
            p3 * b3 + 
            p4 * b4 + 
            p5 * b5 + 
            p6 * b6 + 
            p7 * b7 + 
            p8 * b8 + 
            p9 * b9)
