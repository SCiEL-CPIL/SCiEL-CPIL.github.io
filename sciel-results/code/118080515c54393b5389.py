import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: full quadratic polynomial in normalized coordinates.
    
    The target function is unknown, but we have 10 parameters available.
    We use a full second-order polynomial in 3 variables (normalized to [-1, 1]):
    - 1 (constant)
    - 3 linear terms
    - 3 squared terms
    - 3 interaction terms
    Total: 10 terms, using all 10 parameters.
    
    Normalization: inputs are in [0.5, 2.0], so center=1.25, half_range=0.75.
    This improves conditioning for BFGS optimization.
    """
    p = params
    
    # Normalize to [-1, 1] for better numerical conditioning
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Full quadratic polynomial with 10 coefficients
    # p[0]: constant
    # p[1]: u, p[2]: v, p[3]: w
    # p[4]: u^2, p[5]: v^2, p[6]: w^2
    # p[7]: uv, p[8]: uw, p[9]: vw
    result = (p[0] + p[1]*u + p[2]*v + p[3]*w +
              p[4]*u*u + p[5]*v*v + p[6]*w*w +
              p[7]*u*v + p[8]*u*w + p[9]*v*w)
    
    return result
