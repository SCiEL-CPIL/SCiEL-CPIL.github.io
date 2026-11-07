"""Symbolic regression for a nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physically-motivated nonlinear oscillator model (generalized Duffing).
    
    a = p9 + p0*x + p1*v + p2*x^2 + p3*v^2 + p4*x*v + p5*x^3 + p6*v^3 
        + p7*cos(p8*t) + ... 
    Wait, that's 9 params. Let me restructure.
    
    Actually, let me use:
    a = p0*x + p1*v + p2*x^2 + p3*v^2 + p4*x*v + p5*x^3 + p6*v^3 
        + p7*cos(p8*t + p9)
    
    This gives a Duffing-like model with quadratic and cubic nonlinearities,
    cross terms, and a single periodic forcing term with adjustable frequency and phase.
    
    Inputs are scaled to O(1) for better conditioning.
    """
    # Scale inputs to roughly O(1) magnitude for numerical stability
    # t in [30, 50] -> center at 40, scale by 10
    # x in [-0.2056, 0.1749] -> center at -0.015, scale by 0.4
    # v in [-0.3391, 0.3624] -> center at 0.01, scale by 0.7
    
    ts = (t - 40.0) / 10.0       # O(1)
    xs = (x + 0.015) / 0.4       # O(1)
    vs = (v - 0.01) / 0.7        # O(1)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core nonlinear dynamics
    a = (p0 * xs + 
         p1 * vs + 
         p2 * xs**2 + 
         p3 * vs**2 + 
         p4 * xs * vs + 
         p5 * xs**3 + 
         p6 * vs**3 +
         p7 * np.cos(p8 * ts + p9))
    
    return a
# EVOLVE-BLOCK-END
