"""Parametric limit-cycle oscillator with bounded nonlinearities for symbolic regression."""
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator with bounded nonlinear terms.
    
    Structure: a = -k(x)*x - c(v)*v + F(t) + offset
    where k(x) and c(v) are smooth bounded functions using tanh,
    and F(t) is a dual-frequency periodic drive.
    
    params (10 constants):
        p[0]: drive amplitude 1
        p[1]: drive frequency 1
        p[2]: drive amplitude 2
        p[3]: drive frequency 2
        p[4]: linear spring constant
        p[5]: nonlinear spring coefficient (tanh saturation)
        p[6]: linear damping
        p[7]: nonlinear damping coefficient (tanh saturation)
        p[8]: quadratic spring term
        p[9]: constant offset
    """
    p = params
    
    # Bounded nonlinear spring: k(x) = p[4] + p[5] * tanh(x)
    kx = p[4] + p[5] * np.tanh(x)
    
    # Bounded nonlinear damping: c(v) = p[6] + p[7] * tanh(v)
    cv = p[6] + p[7] * np.tanh(v)
    
    # Dual-frequency periodic drive
    F = p[0] * np.sin(p[1] * t) + p[2] * np.sin(p[3] * t)
    
    # Quadratic spring term (bounded by tanh to prevent explosion)
    quad = p[8] * x * np.tanh(x)
    
    # Combine: a = -k(x)*x - c(v)*v + F + quad + offset
    a = -kx * x - cv * v + F + quad + p[9]
    
    return a
