"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed nonlinear oscillator model using bounded nonlinearities.
    
    Uses sin(x), tanh(x), sin(v), tanh(v) and their interactions to capture
    periodic restoring forces and velocity saturation, avoiding polynomial
    extrapolation instability.
    
    params: array of 10 fitted constants
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinearities for position
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    
    # Bounded nonlinearities for velocity
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Interaction terms
    x_tanh_v = x * tanh_v
    tanh_x_v = tanh_x * v
    
    # Construct the equation
    # p0: linear position (restoring force)
    # p1: sin(x) - periodic restoring
    # p2: tanh(x) - saturating restoring
    # p3: linear velocity (damping)
    # p4: sin(v) - periodic velocity effect
    # p5: tanh(v) - saturating velocity effect
    # p6: x * tanh(v) - position-velocity interaction
    # p7: tanh(x) * v - position-velocity interaction
    # p8: x * v - bilinear interaction
    # p9: constant offset
    
    return (p0 * x + p1 * sin_x + p2 * tanh_x + 
            p3 * v + p4 * sin_v + p5 * tanh_v +
            p6 * x_tanh_v + p7 * tanh_x_v +
            p8 * x * v + p9)
# EVOLVE-BLOCK-END
