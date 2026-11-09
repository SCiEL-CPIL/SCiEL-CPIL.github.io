# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: combines a polynomial/interaction basis (for global structure)
    # with a small set of Radial Basis Functions (for local non-linear features).
    # The model is linear in params, so BFGS can converge reliably to the
    # global minimum of the least-squares objective.
    #
    # Basis functions (10 total):
    # 0: 1 (bias)
    # 1: x
    # 2: y
    # 3: z
    # 4: x*y
    # 5: x*z
    # 6: y*z
    # 7: x*y*z
    # 8: RBF centered at (0.5, 0.5, 0.5) with gamma=1.0
    # 9: RBF centered at (2.0, 2.0, 2.0) with gamma=1.0
    #
    # The two RBFs are placed at opposite corners to capture potential
    # non-polynomial behavior near the boundaries of the domain [0.5, 2]^3.
    # gamma=1.0 provides a reasonable bandwidth: the kernel decays to ~0.001
    # at the maximum distance (~sqrt(6.75) ~ 2.6), ensuring locality.
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    
    # Polynomial/interaction features
    f0 = np.ones(n, dtype=np.float64)
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = x * y * z
    
    # RBF features
    # Center 1: (0.5, 0.5, 0.5)
    dx1 = x - 0.5
    dy1 = y - 0.5
    dz1 = z - 0.5
    dist_sq1 = dx1*dx1 + dy1*dy1 + dz1*dz1
    gamma = 1.0
    f8 = np.exp(-gamma * dist_sq1)
    
    # Center 2: (2.0, 2.0, 2.0)
    dx2 = x - 2.0
    dy2 = y - 2.0
    dz2 = z - 2.0
    dist_sq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
    f9 = np.exp(-gamma * dist_sq2)
    
    # Linear combination
    pred = (p[0] * f0 + 
            p[1] * f1 + 
            p[2] * f2 + 
            p[3] * f3 + 
            p[4] * f4 + 
            p[5] * f5 + 
            p[6] * f6 + 
            p[7] * f7 + 
            p[8] * f8 + 
            p[9] * f9)
            
    return pred
# EVOLVE-BLOCK-END
