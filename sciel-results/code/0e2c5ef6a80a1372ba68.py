"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Uses a physics-informed basis of smooth, bounded nonlinearities:
    - sin(x), tanh(x) for restoring force (periodic + saturation)
    - v, tanh(v), sin(v) for damping (linear + nonlinear velocity dependence)
    - Cross-coupling terms between position and velocity
    
    params[0] : coefficient for sin(x)
    params[1] : coefficient for tanh(x)
    params[2] : coefficient for x
    params[3] : coefficient for v
    params[4] : coefficient for tanh(v)
    params[5] : coefficient for sin(v)
    params[6] : coefficient for sin(x * v)
    params[7] : coefficient for tanh(x) * v
    params[8] : coefficient for x * sin(v)
    params[9] : constant offset
    """
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Restoring force terms
    term_sin_x = params[0] * sin_x
    term_tanh_x = params[1] * tanh_x
    term_x = params[2] * x
    
    # Damping terms
    term_v = params[3] * v
    term_tanh_v = params[4] * tanh_v
    term_sin_v = params[5] * sin_v
    
    # Cross-coupling terms
    term_sin_xv = params[6] * np.sin(x * v)
    term_tanhx_v = params[7] * tanh_x * v
    term_x_sin_v = params[8] * x * sin_v
    
    # Constant
    const = params[9]
    
    return term_sin_x + term_tanh_x + term_x + term_v + term_tanh_v + term_sin_v + term_sin_xv + term_tanhx_v + term_x_sin_v + const
# EVOLVE-BLOCK-END
