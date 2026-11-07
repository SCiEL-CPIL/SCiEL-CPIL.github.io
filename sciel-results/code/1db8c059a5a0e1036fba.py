"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (global smooth trend) + 5 Gaussian RBFs (local features).
    
    The first 5 params weight Chebyshev polynomials T_0..T_4 for global smoothness.
    The last 5 params weight Gaussian RBFs centered at fixed locations with fixed widths.
    
    This hybrid approach:
    - Chebyshev terms capture smooth, global structure (polynomial-like, trig, etc.)
    - RBF terms capture local features, discontinuities, or sharp transitions
    - The combination is still linear in params, so BFGS converges to global optimum
    
    RBF centers are fixed at equally spaced points in [-2, 2] with moderate width.
    """
    # Global component: Chebyshev polynomials T_0 through T_4
    t = x / 2.0  # Map [-2,2] to [-1,1]
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Local component: Gaussian RBFs
    # Fixed centers at -2, -1, 0, 1, 2
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # Fixed width: sigma = 0.8 (moderate locality, covers domain well)
    sigma = 0.8
    
    result = np.zeros_like(x)
    
    # Add Chebyshev contributions (params[0:5])
    result += params[0] * T0
    result += params[1] * T1
    result += params[2] * T2
    result += params[3] * T3
    result += params[4] * T4
    
    # Add RBF contributions (params[5:10])
    for i in range(5):
        diff = x - centers[i]
        rbf = np.exp(-0.5 * (diff / sigma) ** 2)
        result += params[5 + i] * rbf
    
    return result
# EVOLVE-BLOCK-END
