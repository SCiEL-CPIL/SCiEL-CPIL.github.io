"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a tensor-product Fourier basis.
    
    Basis functions (10 total):
    0. 1                          (constant)
    1. sin(pi*x)                  (fundamental frequency in x)
    2. sin(pi*y)                  (fundamental frequency in y)
    3. cos(2*pi*x)                (second harmonic in x)
    4. cos(2*pi*y)                (second harmonic in y)
    5. sin(2*pi*x)                (second harmonic sine in x)
    6. sin(2*pi*y)                (second harmonic sine in y)
    7. x*y                        (bilinear interaction)
    8. sin(pi*x)*sin(pi*y)        (product of fundamental sines)
    9. cos(pi*x)*cos(pi*y)        (product of fundamental cosines)
    
    Fourier basis is orthogonal on [-1,1] with appropriate scaling,
    providing numerical stability for BFGS optimization.
    """
    # Precompute basis functions
    sin_pi_x = np.sin(np.pi * x)
    sin_pi_y = np.sin(np.pi * y)
    cos_2pi_x = np.cos(2.0 * np.pi * x)
    cos_2pi_y = np.cos(2.0 * np.pi * y)
    sin_2pi_x = np.sin(2.0 * np.pi * x)
    sin_2pi_y = np.sin(2.0 * np.pi * y)
    xy = x * y
    sin_pi_x_sin_pi_y = sin_pi_x * sin_pi_y
    cos_pi_x = np.cos(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    cos_pi_x_cos_pi_y = cos_pi_x * cos_pi_y
    
    # Construct prediction as linear combination
    pred = (
        params[0] * 1.0 +
        params[1] * sin_pi_x +
        params[2] * sin_pi_y +
        params[3] * cos_2pi_x +
        params[4] * cos_2pi_y +
        params[5] * sin_2pi_x +
        params[6] * sin_2pi_y +
        params[7] * xy +
        params[8] * sin_pi_x_sin_pi_y +
        params[9] * cos_pi_x_cos_pi_y
    )
    
    return pred
# EVOLVE-BLOCK-END
