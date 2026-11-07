"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator, approximated using a 2D Fourier series expansion.
    
    This approach leverages the periodic nature of oscillator dynamics by using
    sine and cosine basis functions. This is distinct from polynomial expansions
    and often captures oscillatory behavior more efficiently.
    
    Basis functions (up to 2nd harmonic in x and v, plus one mixed term):
    1. 1
    2. sin(pi * u)
    3. cos(pi * u)
    4. sin(pi * w)
    5. cos(pi * w)
    6. sin(2 * pi * u)
    7. cos(2 * pi * u)
    8. sin(2 * pi * w)
    9. cos(2 * pi * w)
    10. sin(pi * u) * sin(pi * w)
    
    where u and w are normalized x and v to [-1, 1].
    """
    # Input ranges from the task description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Normalize inputs to [-1, 1]
    x_center = (x_max + x_min) / 2.0
    x_range = x_max - x_min
    v_center = (v_max + v_min) / 2.0
    v_range = v_max - v_min
    
    # Avoid division by zero
    if x_range == 0: x_range = 1e-8
    if v_range == 0: v_range = 1e-8
    
    u = (2.0 * x - x_center) / x_range
    w = (2.0 * v - v_center) / v_range
    
    # Clip to ensure numerical stability (though sin/cos are periodic, 
    # clipping prevents huge arguments if inputs are far out of range)
    u = np.clip(u, -10.0, 10.0)
    w = np.clip(w, -10.0, 10.0)
    
    # Compute Fourier basis functions
    p = params
    
    b0 = np.ones_like(u)
    
    b1 = np.sin(np.pi * u)
    b2 = np.cos(np.pi * u)
    
    b3 = np.sin(np.pi * w)
    b4 = np.cos(np.pi * w)
    
    b5 = np.sin(2.0 * np.pi * u)
    b6 = np.cos(2.0 * np.pi * u)
    
    b7 = np.sin(2.0 * np.pi * w)
    b8 = np.cos(2.0 * np.pi * w)
    
    b9 = np.sin(np.pi * u) * np.sin(np.pi * w)
    
    # Linear combination with parameters
    return p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 + \
           p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9
# EVOLVE-BLOCK-END
