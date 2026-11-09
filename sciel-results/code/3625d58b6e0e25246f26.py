"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: high-order Chebyshev polynomials + Gaussian RBFs.
    
    The basis is linear in params, so BFGS solves a convex least-squares problem.
    Chebyshev polynomials provide global smooth approximation.
    Gaussian RBFs (with fixed, well-spread centers) provide local flexibility.
    
    params[0:6]  -> Chebyshev coefficients T_0..T_5 (scaled)
    params[6:10] -> Gaussian RBF weights (4 centers)
    """
    # Scale x from [-2, 2] to z in [-1, 1] for stable Chebyshev evaluation
    z = x / 2.0
    
    # Chebyshev polynomials T_0 through T_5
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * T1 - T0
    T3 = 2.0 * z * T2 - T1
    T4 = 2.0 * z * T3 - T2
    T5 = 2.0 * z * T4 - T3
    
    # Gaussian RBFs with fixed centers spread across [-2, 2]
    # Centers: -1.5, -0.5, 0.5, 1.5
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    # Width: 0.8 gives good local coverage
    width = 0.8
    
    # Compute RBFs: sum of 4 Gaussians weighted by params[6:10]
    rbf_sum = np.zeros_like(x)
    for i in range(4):
        rbf_sum += params[6 + i] * np.exp(-((x - centers[i]) ** 2) / (2.0 * width ** 2))
    
    # Combine
    result = (params[0] * T0 + params[1] * T1 + params[2] * T2 + 
              params[3] * T3 + params[4] * T4 + params[5] * T5 + rbf_sum)
    
    return result
# EVOLVE-BLOCK-END
