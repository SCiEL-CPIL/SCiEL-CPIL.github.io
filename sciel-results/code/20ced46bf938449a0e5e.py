"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)

    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params

    # Normalize inputs to [-1, 1] range
    # Domain is [0.5, 2.0]
    # Center = 1.25, Half-Range = 0.75
    x_c = (x - 1.25) / 0.75
    y_c = (y - 1.25) / 0.75
    z_c = (z - 1.25) / 0.75

    # Construct a Full Quadratic Basis (10 terms)
    # This spans the space of all polynomials of degree <= 2 in x_c, y_c, z_c
    # f = p0 + p1*x + p2*y + p3*z + p4*x^2 + p5*y^2 + p6*z^2 + p7*xy + p8*xz + p9*yz
    
    f0 = np.ones_like(x)
    f1 = x_c
    f2 = y_c
    f3 = z_c
    f4 = x_c * x_c
    f5 = y_c * y_c
    f6 = z_c * z_c
    f7 = x_c * y_c
    f8 = x_c * z_c
    f9 = y_c * z_c

    # Combine
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
            
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
# EVOLVE-BLOCK-END
