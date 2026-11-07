import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a scaled cubic polynomial basis.
    
    Inputs are scaled to improve numerical conditioning for the BFGS optimizer.
    Basis:
    1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    """
    # Scaling factors based on training ranges to normalize inputs
    # x in [-0.57, 0.53], v in [-0.18, 0.28]
    x_scale = 0.6
    v_scale = 0.3
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Construct basis matrix
    # Using np.column_stack for clarity and efficiency
    phi = np.column_stack((
        np.ones_like(xs),
        xs,
        vs,
        xs**2,
        xs * vs,
        vs**2,
        xs**3,
        xs**2 * vs,
        xs * vs**2,
        vs**3
    ))
    
    return phi @ params
