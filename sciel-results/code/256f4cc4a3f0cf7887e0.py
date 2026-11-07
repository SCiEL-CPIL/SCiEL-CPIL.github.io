import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed basis of bounded nonlinearities.
    
    The basis includes:
    - Restoring forces: sin(x), tanh(x), x, x^3
    - Damping forces: sin(v), tanh(v), v, v^3
    - Coupling: x*v
    - Bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute basis functions
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    x3 = x * x * x
    
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    v3 = v * v * v
    
    xv = x * v

    # Linear combination of basis functions
    return (
        p0 * sin_x
        + p1 * tanh_x
        + p2 * x
        + p3 * x3
        + p4 * sin_v
        + p5 * tanh_v
        + p6 * v
        + p7 * v3
        + p8 * xv
        + p9
    )
