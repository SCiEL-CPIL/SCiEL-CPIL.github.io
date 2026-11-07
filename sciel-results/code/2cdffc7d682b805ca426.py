"""
Hybrid Chebyshev-Polynomial + Adaptive Gaussian RBF basis.

Structure:
- 5 Chebyshev polynomial terms (T0..T4) mapped to [-1,1] for numerical stability
- 5 Gaussian RBF kernels with fixed centers at {-3, -1.5, 0, 1.5, 3}

Key improvements over the current RBF+raw polynomial approach:
1. Chebyshev polynomials are orthogonal on [-1,1], providing better conditioning
   than raw powers and capturing global trends more efficiently.
2. Gaussian RBFs provide local flexibility for features not captured by the
   global polynomial trend.
3. All terms are linear in params, so BFGS solves a convex problem reliably.

The combination gives:
- Global smooth trends via Chebyshev basis (better conditioned than raw powers)
- Local bumps/features via Gaussian RBFs
- 10 parameters total, all linear in output
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev + Gaussian RBF basis function.
    
    Parameters:
        x: array of input values
        params: array of 10 parameters
            params[0:5] - coefficients for Chebyshev terms T0..T4
            params[5:10] - coefficients for RBF kernels at centers {-3, -1.5, 0, 1.5, 3}
    
    Returns:
        Array of predictions, one per input point
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    xp = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(xp)
    T1 = xp
    T2 = 2.0 * xp * xp - 1.0
    T3 = 4.0 * xp**3 - 3.0 * xp
    T4 = 8.0 * xp**4 - 8.0 * xp**2 + 1.0
    
    # Polynomial part (Chebyshev basis)
    y = (p[0] * T0 +
         p[1] * T1 +
         p[2] * T2 +
         p[3] * T3 +
         p[4] * T4)
    
    # RBF basis (5 Gaussian kernels)
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0  # Gaussian width
    
    for i, c in enumerate(centers):
        y += p[5 + i] * np.exp(-0.5 * ((x - c) / sigma)**2)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
