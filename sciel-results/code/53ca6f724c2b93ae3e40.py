"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Product-of-sums basis: f(x,y,z) = prod_{i=1}^{3} (a_i + b_i*x + c_i*y + d_i*z)
    # This gives 10 parameters total (3 terms * 3 coeffs + 1 implicit constant from expansion)
    # Actually let's use: f = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) * (p8 + p9*x + p10*y + p11*z)
    # But we only have 10 params. Let me restructure.
    # 
    # With 10 params, use 2 full linear factors (4 params each = 8) plus 2 extra:
    # f = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) + p8*x + p9*y
    # 
    # Actually, let's do a cleaner design:
    # f = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) + p8*(x*y) + p9*(y*z)
    # This captures bilinear interactions plus a pure product of two hyperplanes.
    
    p = params
    n = len(x)
    
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Two linear factors
    f1 = p[0] + p[1]*x + p[2]*y + p[3]*z
    f2 = p[4] + p[5]*x + p[6]*y + p[7]*z
    
    # Product of sums
    result = f1 * f2 + p[8]*x*y + p[9]*y*z
    
    return result
# EVOLVE-BLOCK-END
