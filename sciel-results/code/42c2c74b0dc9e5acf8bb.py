"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    # Clamp to [-1, 1] to avoid numerical issues with arccos if x slightly outside (though x is uniform [-3,3])
    u_clamped = np.clip(u, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_n(u) = 2u T_{n-1}(u) - T_{n-2}(u)
    
    t0 = np.ones_like(u_clamped)
    t1 = u_clamped.copy()
    
    result = params[0] * t0 + params[1] * t1
    
    if len(params) > 2:
        t_prev2 = t0
        t_prev1 = t1
        for k in range(2, 10):
            # T_k = 2*u*T_{k-1} - T_{k-2}
            t_curr = 2.0 * u_clamped * t_prev1 - t_prev2
            result += params[k] * t_curr
            t_prev2 = t_prev1
            t_prev1 = t_curr
            
    return result
# EVOLVE-BLOCK-END
