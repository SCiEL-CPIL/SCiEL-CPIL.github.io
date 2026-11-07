import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 10-term polynomial basis in 3D:
    {1, x, y, z, x*y, x*z, y*z, x^2, y^2, z^2}
    
    This full interaction + quadratic basis captures additive structure,
    pairwise interactions, and curvature without forcing separability.
    All inputs are in [0.5, 2], so no normalization needed for stability.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 10 basis functions: constant, linear, pairwise interactions, quadratics
    result = (p[0] * 1.0
             + p[1] * x
             + p[2] * y
             + p[3] * z
             + p[4] * x * y
             + p[5] * x * z
             + p[6] * y * z
             + p[7] * x * x
             + p[8] * y * y
             + p[9] * z * z)
    
    return result
