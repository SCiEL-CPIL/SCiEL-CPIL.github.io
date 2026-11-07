"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a driven nonlinear oscillator.
    
    Structural Change:
    We use a richer basis set that includes:
    1. Sinusoidal time terms (sin, cos) for periodic driving.
    2. Polynomial terms in x and v up to degree 3.
    3. Cross terms between x and v.
    4. Parametric excitation terms (x^2 * sin/cos, v^2 * sin/cos).
    5. Scaled inputs to improve conditioning.
    
    This gives the fitter more degrees of freedom to capture complex dynamics
    including limit cycles, parametric resonance, and nonlinear coupling.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to O(1) for better conditioning
    # x in [-0.34, 0.36], v in [-0.34, 0.36]
    # Scale by 2 to map to approx [-0.7, 0.7]
    xn = x * 2.0
    vn = v * 2.0
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Time terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Build the model with diverse basis functions
    # p0: constant offset
    # p1: linear in x (restoring force)
    # p2: cubic in x (Duffing nonlinearity)
    # p3: linear in v (damping)
    # p4: cubic in v (nonlinear damping)
    # p5: x^2 * v (Van der Pol term)
    # p6: x * v (cross coupling)
    # p7: sin(t) (periodic drive)
    # p8: cos(t) (quadrature drive)
    # p9: x^2 (quadratic position)
    
    a = p0
    a = a + p1 * xn
    a = a + p2 * xn3
    a = a + p3 * vn
    a = a + p4 * vn3
    a = a + p5 * xn2 * vn
    a = a + p6 * xn * vn
    a = a + p7 * sin_t
    a = a + p8 * cos_t
    a = a + p9 * xn2
    
    return a
# EVOLVE-BLOCK-END
