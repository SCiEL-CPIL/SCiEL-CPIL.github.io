"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Uses a physics-informed basis of smooth, bounded nonlinearities:
    - sin(x), tanh(x), sin(x)*tanh(x) for restoring force
    - tanh(v), sin(v) for damping
    - Cross-coupling terms using products of these bounded functions
    
    params[0] : coefficient for sin(x)
    params[1] : coefficient for tanh(x)
    params[2] : coefficient for sin(x) * tanh(x)
    params[3] : coefficient for tanh(v)
    params[4] : coefficient for sin(v)
    params[5] : coefficient for tanh(x) * tanh(v)
    params[6] : coefficient for sin(x) * sin(v)
    params[7] : coefficient for tanh(x) * v
    params[8] : coefficient for x * tanh(v)
    params[9] : constant offset
    """
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Restoring force terms
    term_0 = params[0] * sin_x
    term_1 = params[1] * tanh_x
    term_2 = params[2] * sin_x * tanh_x
    
    # Damping terms
    term_3 = params[3] * tanh_v
    term_4 = params[4] * sin_v
    
    # Cross-coupling terms
    term_5 = params[5] * tanh_x * tanh_v
    term_6 = params[6] * sin_x * sin_v
    term_7 = params[7] * tanh_x * v
    term_8 = params[8] * x * tanh_v
    
    # Constant
    const = params[9]
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + const
# EVOLVE-BLOCK-END
