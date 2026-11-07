"""Nonlinear oscillator model using physics-informed smooth basis functions."""
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear damped oscillator with driving force.
    
    Uses smooth, bounded nonlinearities (sin, tanh) to capture:
    - Periodic restoring force (sin terms in x)
    - Saturation of nonlinear effects (tanh terms)
    - Velocity-dependent damping and nonlinear velocity effects
    - Interaction between position and velocity
    
    params: array of length 10
      p0: linear restoring force coefficient
      p1: cubic-like restoring (sin(x) approximates odd nonlinearity)
      p2: higher-order restoring (sin(2x) or sin(x)^3 component)
      p3: linear damping coefficient
      p4: nonlinear velocity effect (tanh(v))
      p5: velocity squared effect (v^2)
      p6: interaction x*v
      p7: interaction x^2*v
      p8: interaction x*tanh(v)
      p9: constant offset / driving term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth nonlinear position terms
    sin_x = np.sin(x)
    sin_2x = np.sin(2.0 * x)
    
    # Smooth nonlinear velocity terms
    tanh_v = np.tanh(v)
    v2 = v * v
    
    # Interaction terms
    xv = x * v
    x2v = x * x * v
    x_tanh_v = x * tanh_v
    
    # Combine all terms
    result = (
        p0 * x
        + p1 * sin_x
        + p2 * sin_2x
        + p3 * v
        + p4 * tanh_v
        + p5 * v2
        + p6 * xv
        + p7 * x2v
        + p8 * x_tanh_v
        + p9
    )
    
    return result
