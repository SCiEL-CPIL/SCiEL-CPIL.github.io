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
    
    # Bounded nonlinear stiffness with quadratic correction:
    # k(x) = p0 + p1 * tanh(p2 * x) + 0.5 * p1 * tanh(p2 * x)^2
    tx = np.tanh(p2 * x)
    kx = p0 + p1 * tx + 0.5 * p1 * tx * tx
    
    # Bounded nonlinear damping with quadratic correction:
    # c(v) = p3 + p4 * tanh(p5 * v) + 0.5 * p4 * tanh(p5 * v)^2
    tv = np.tanh(p5 * v)
    cv = p3 + p4 * tv + 0.5 * p4 * tv * tv
    
    # Driving force: F(t) = p6 * sin(p7 * t + p8) + p9 * t
    F = p6 * np.sin(p7 * t + p8) + p9 * t
    
    # Cross-coupling term: -0.1 * p4 * x * v (small, bounded by tanh structure)
    cross = -0.1 * p4 * x * v
    
    # Acceleration: a = -k(x)*x - c(v)*v + cross + F(t)
    a = -kx * x - cv * v + cross + F
    
    return a
# EVOLVE-BLOCK-END
