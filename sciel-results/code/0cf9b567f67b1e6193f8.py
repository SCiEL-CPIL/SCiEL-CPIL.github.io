"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Uses a physics-informed basis of smooth, bounded nonlinearities to capture
    periodic restoring forces and saturation effects without ill-conditioning
    from high-degree polynomials.
    
    Basis:
    - Restoring: sin(x), sin(2x), sin(3x), tanh(x)
    - Damping: v, tanh(v)
    - Coupling: x*v, sin(x)*v, tanh(x)*v
    - Constant
    
    params[0] : coefficient for sin(x)
    params[1] : coefficient for sin(2*x)
    params[2] : coefficient for sin(3*x)
    params[3] : coefficient for tanh(x)
    params[4] : coefficient for v
    params[5] : coefficient for tanh(v)
    params[6] : coefficient for x * v
    params[7] : coefficient for sin(x) * v
    params[8] : coefficient for tanh(x) * v
    params[9] : constant offset
    """
    sin_x = np.sin(x)
    sin_2x = np.sin(2 * x)
    sin_3x = np.sin(3 * x)
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # Restoring force terms (bounded non-linearities)
    term_sin_x = params[0] * sin_x
    term_sin_2x = params[1] * sin_2x
    term_sin_3x = params[2] * sin_3x
    term_tanh_x = params[3] * tanh_x
    
    # Damping terms
    term_v = params[4] * v
    term_tanh_v = params[5] * tanh_v
    
    # Cross-coupling terms
    term_x_v = params[6] * x * v
    term_sin_x_v = params[7] * sin_x * v
    term_tanh_x_v = params[8] * tanh_x * v
    
    # Constant
    const = params[9]
    
    return (term_sin_x + term_sin_2x + term_sin_3x + term_tanh_x + 
            term_v + term_tanh_v + 
            term_x_v + term_sin_x_v + term_tanh_x_v + 
            const)
# EVOLVE-BLOCK-END
