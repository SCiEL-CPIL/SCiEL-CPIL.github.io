"""
Hybrid Chebyshev + Multi-scale Gaussian RBF basis for symbolic regression.

Structure:
- 4 Chebyshev polynomial terms (T0..T3) mapped to [-1,1] for numerical stability
- 6 Gaussian RBF kernels with fixed centers at {-3, -2, -1, 0, 1, 2}
  (asymmetric coverage of [-3,3] with spacing 1.0 and sigma=0.8)

This combines the global trend capture of orthogonal Chebyshev polynomials
with local flexibility from multiple Gaussian bumps. The model is linear in
parameters, so BFGS solves a convex problem reliably from the all-ones
starting point.

Key design choices:
1. Chebyshev polynomials are orthogonal on [-1,1], providing better
   conditioning than raw powers and capturing global trends efficiently.
2. Six Gaussian RBFs with spacing 1.0 and moderate width (sigma=0.8)
   provide local flexibility to capture features not in the polynomial trend.
3. The asymmetric center placement {-3,-2,-1,0,1,2} covers the domain well
   with slightly more resolution on the left side where x=-3 is an endpoint.
4. All terms are linear in params, ensuring a convex optimization landscape.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev + Multi-scale Gaussian RBF basis function.
    
    Parameters:
        x: array of input values
        params: array of 10 parameters
            params[0:4] - coefficients for Chebyshev terms (T0, T1, T2, T3)
            params[4:10] - coefficients for RBF kernels at centers {-3,-2,-1,0,1,2}
    
    Returns:
        Array of predictions, one per input point
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    xp = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_3
    T0 = np.ones_like(xp)
    T1 = xp
    T2 = 2.0 * xp * xp - 1.0
    T3 = 4.0 * xp**3 - 3.0 * xp
    
    # Polynomial part (4 Chebyshev terms)
    y = (p[0] * T0 +
         p[1] * T1 +
         p[2] * T2 +
         p[3] * T3)
    
    # RBF part (6 Gaussian kernels)
    centers = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    sigma = 0.8  # Gaussian width
    
    for i, c in enumerate(centers):
        y += p[4 + i] * np.exp(-0.5 * ((x - c) / sigma)**2)
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
