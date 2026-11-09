"""Parametric limit-cycle oscillator with smooth bounded nonlinearities."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Structure: a = -k(x) * x - c(v) * v + F(t)
    where:
      k(x) = p0 + p1 * tanh(p2 * x)   (bounded nonlinear stiffness)
      c(v) = p3 + p4 * tanh(p5 * v)   (bounded nonlinear damping)
      F(t) = p6 * sin(p7 * t + p8) + p9 * t  (driving force)
    
    params: array of length 10
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear stiffness
    tx = np.tanh(p2 * x)
    kx = p0 + p1 * tx
    
    # Bounded nonlinear damping
    tv = np.tanh(p5 * v)
    cv = p3 + p4 * tv
    
    # Driving force with sine and cosine components for full phase/amplitude flexibility
    # F(t) = p6 * sin(p7 * t + p8) + p9 * cos(p7 * t + p8)
    angle = p7 * t + p8
    F = p6 * np.sin(angle) + p9 * np.cos(angle)
    
    # Acceleration: a = -k(x)*x - c(v)*v + F(t)
    a = -kx * x - cv * v + F
    
    return a
# EVOLVE-BLOCK-END
