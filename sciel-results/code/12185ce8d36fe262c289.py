"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a global start time to monitor elapsed time if this were a loop,
# but here it's a single function call. The "guard" is ensuring vectorized,
# low-latency operations to prevent the optimizer from timing out.
# Since equation is called many times, keeping it simple is key.

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1]
    # Inputs are in [0.5, 2.0]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Shifted Legendre Polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    # P3(t) = 0.125 * (5t^3 - 3t)
    
    # Precompute powers for efficiency
    u2 = u * u
    v2 = v * v
    w2 = w * w
    
    u3 = u2 * u
    v3 = v2 * v
    w3 = w2 * w
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(u)
    
    # 2-4. Linear (P1)
    b1 = u
    b2 = v
    b3 = w
    
    # 5-7. Quadratic (P2)
    b4 = 0.5 * (3.0 * u2 - 1.0)
    b5 = 0.5 * (3.0 * v2 - 1.0)
    b6 = 0.5 * (3.0 * w2 - 1.0)
    
    # 8-10. Cubic (P3)
    b7 = 0.125 * (5.0 * u3 - 3.0 * u)
    b8 = 0.125 * (5.0 * v3 - 3.0 * v)
    b9 = 0.125 * (5.0 * w3 - 3.0 * w)
    
    # Combine
    # This basis captures additive cubic structure with orthogonal, well-conditioned terms.
    # This mirrors the high-scoring inspiration's cubic pure-power structure but uses
    # Legendre polynomials for better numerical conditioning during BFGS fitting.
    return (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
# EVOLVE-BLOCK-END
