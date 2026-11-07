"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Polynomial-Trigonometric model for a driven nonlinear oscillator.
    
    Combines standard Duffing-like state nonlinearities with a fundamental 
    periodic driving term and low-order time trends. This approach family 
    (Trigonometric + Polynomial) is distinct from the pure polynomial basis 
    of the seed program and better captures periodic forcing.
    
    Terms:
    - p0: Constant offset
    - p1*x: Linear restoring
    - p2*v: Linear damping
    - p3*x^3: Cubic stiffness (Duffing)
    - p4*x*v: Nonlinear damping
    - p5*v^2: Quadratic velocity effect
    - p6*cos(t): Periodic drive (cosine)
    - p7*sin(t): Periodic drive (sine)
    - p8*t: Linear time trend
    - p9*t^2: Quadratic time trend
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # State dynamics (Nonlinear)
    term_x = p1 * x
    term_v = p2 * v
    term_x3 = p3 * (x ** 3)
    term_xv = p4 * x * v
    term_v2 = p5 * (v ** 2)
    
    # Time dynamics (Periodic + Trend)
    term_cos = p6 * np.cos(t)
    term_sin = p7 * np.sin(t)
    term_t = p8 * t
    term_t2 = p9 * (t ** 2)
    
    return p0 + term_x + term_v + term_x3 + term_xv + term_v2 + term_cos + term_sin + term_t + term_t2
# EVOLVE-BLOCK-END
