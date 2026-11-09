import numpy as np
import time

def _chebyshev_T(x, n):
    """Compute Chebyshev polynomial of first kind of degree n."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        Tn_1 = x.copy()
        Tn_2 = np.ones_like(x)
        for _ in range(2, n + 1):
            Tn = 2.0 * x * Tn_1 - Tn_2
            Tn_2 = Tn_1
            Tn_1 = Tn
        return Tn_1

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: f(x) = N(x) / D(x)
    where N(x) = sum_{i=0}^{4} a_i * T_i(x)
         D(x) = 1 + sum_{j=1}^{4} b_j * T_j(x)
    
    Params:
    0: a0
    1: a1
    2: a2
    3: a3
    4: a4
    5: b1
    6: b2
    7: b3
    8: b4
    9: scale (multiplicative scale for stability)
    """
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    b1, b2, b3, b4 = params[5], params[6], params[7], params[8]
    scale = params[9]
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(x)
    T1 = x.copy()
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    T4 = 8.0 * x**4 - 8.0 * x**2 + 1.0
    
    # Numerator
    N = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: use tanh to constrain b_j contributions, keeping D near 1
    # This prevents D from approaching zero and improves BFGS convergence
    d1 = np.tanh(b1)
    d2 = np.tanh(b2)
    d3 = np.tanh(b3)
    d4 = np.tanh(b4)
    
    D = 1.0 + d1 * T1 + d2 * T2 + d3 * T3 + d4 * T4
    
    # Ensure D stays away from zero with a soft floor
    # Use a larger epsilon to keep the function well-behaved for BFGS
    eps = 1e-6
    D_safe = np.where(np.abs(D) < eps, np.sign(D) * eps + (D == 0) * eps, D)
    
    y = N / D_safe
    y = y * scale
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
