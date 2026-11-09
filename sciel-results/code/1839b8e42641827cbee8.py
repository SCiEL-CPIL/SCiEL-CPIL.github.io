"""Parametric limit-cycle oscillator model with bounded nonlinearities."""
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model for a nonlinear oscillator with bounded nonlinearities:
    a = -k(x)*x - c(v)*v + F(t)
    
    where:
      k(x) = 1 + p1*tanh(x) + p2*tanh(x^2)   (bounded nonlinear spring)
      c(v) = 1 + p3*tanh(v)                    (bounded nonlinear damping)
      F(t) = p4*sin(t) + p5*cos(t)             (periodic driving)
    
    Uses 6 parameters. Remaining 4 are available but not used in this form.
    """
    # Bounded nonlinear spring stiffness
    tanh_x = np.tanh(x)
    tanh_x2 = np.tanh(x * x)
    k_x = 1.0 + params[1] * tanh_x + params[2] * tanh_x2
    
    # Bounded nonlinear damping
    tanh_v = np.tanh(v)
    c_v = 1.0 + params[3] * tanh_v
    
    # Periodic driving force
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    F_t = params[4] * sin_t + params[5] * cos_t
    
    # Acceleration
    a = -k_x * x - c_v * v + F_t
    
    # Add a small constant offset and higher-order bounded terms using remaining params
    a = a + params[0] + params[6] * np.tanh(x * v) + params[7] * np.tanh(t) * np.tanh(x)
    
    return a
