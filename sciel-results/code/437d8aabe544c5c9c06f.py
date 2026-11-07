"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with parametric forcing.
    
    Model: a = c1*x + c2*v + c3*x^3 + c4*v^3 + c5*cos(w*t)*x + c6*sin(w*t) + c7*x^2 + c8*v^2 + c9*x*v + c10
    
    Uses parametric excitation (cos(w*t)*x) which is common in nonlinear oscillators,
    along with standard Duffing terms. Time is centered for numerical stability.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center time for better numerical conditioning
    t_c = t - 40.0  # mid-point of [30, 50]
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Use a base frequency that's likely in the driving range
    # For t in [30,50], a frequency of ~1-5 rad/s would give several cycles
    w = 2.0
    cos_wt = np.cos(w * t_c)
    sin_wt = np.sin(w * t_c)
    
    result = (
        p0 * x +
        p1 * v +
        p2 * x3 +
        p3 * v3 +
        p4 * cos_wt * x +
        p5 * sin_wt +
        p6 * x2 +
        p7 * v2 +
        p8 * x * v +
        p9
    )
    
    return result
# EVOLVE-BLOCK-END
