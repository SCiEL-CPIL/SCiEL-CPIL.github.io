"""
Structural change: state-space representation with time-varying coefficients
derived from a low-order Fourier series in t. This allows the model to capture
periodic changes in effective stiffness and damping without high-degree polynomials.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with time-varying coefficients.
    
    Model: a(t) = k(t) * x + c(t) * v + f(t)
    where k(t), c(t), f(t) are low-order Fourier series in t.
    
    With 10 params, we allocate:
    - k(t) = p0 + p1*cos(t) + p2*sin(t)   (3 params: time-varying stiffness)
    - c(t) = p3 + p4*cos(t) + p5*sin(t)   (3 params: time-varying damping)
    - f(t) = p6 + p7*cos(t) + p8*sin(t)   (3 params: time-varying forcing)
    - p9: nonlinear correction term coefficient for x*v coupling
    
    Total: 10 parameters
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric basis
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    # Time-varying stiffness: k(t) = p0 + p1*cos(t) + p2*sin(t)
    k_t = p0 + p1 * cos_t + p2 * sin_t
    
    # Time-varying damping: c(t) = p3 + p4*cos(t) + p5*sin(t)
    c_t = p3 + p4 * cos_t + p5 * sin_t
    
    # Time-varying forcing: f(t) = p6 + p7*cos(t) + p8*sin(t)
    f_t = p6 + p7 * cos_t + p8 * sin_t
    
    # Nonlinear coupling term
    coupling = p9 * x * v
    
    # Full model: a = k(t)*x + c(t)*v + f(t) + coupling
    result = k_t * x + c_t * v + f_t + coupling
    
    return result
# EVOLVE-BLOCK-END
