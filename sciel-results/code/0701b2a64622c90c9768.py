import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a hybrid basis:
    - Radial/angular terms (r^2, r^2 cos2θ, r^2 sin2θ) for isotropic/anisotropic features
    - Higher-order Chebyshev tensor products for directional features
    - Sine/cosine terms for periodic features
    
    10 parameters total.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    p = params
    
    # Radial and angular components
    r2 = x * x + y * y
    r4 = r2 * r2
    cos2theta = x * x - y * y  # proportional to cos(2*theta)
    sin2theta = 2.0 * x * y    # proportional to sin(2*theta)
    
    # Chebyshev polynomials for higher-order tensor products
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    T4_x = 8.0 * x**4 - 8.0 * x * x + 1.0
    
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    T4_y = 8.0 * y**4 - 8.0 * y * y + 1.0
    
    # Basis functions:
    # 0: constant (T0*T0)
    # 1: linear x
    # 2: linear y
    # 3: radial r^2
    # 4: anisotropic cos(2θ) component: x^2 - y^2
    # 5: anisotropic sin(2θ) component: xy
    # 6: cubic x
    # 7: cubic y
    # 8: quartic radial r^4
    # 9: mixed quartic x^2*y^2
    
    pred = (p[0] * np.ones_like(x) +
            p[1] * T1_x +
            p[2] * T1_y +
            p[3] * r2 +
            p[4] * cos2theta +
            p[5] * sin2theta +
            p[6] * T3_x +
            p[7] * T3_y +
            p[8] * r4 +
            p[9] * (x * x) * (y * y))
    
    return pred
