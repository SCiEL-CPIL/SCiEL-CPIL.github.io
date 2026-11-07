# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
    
    # Strategy: Chebyshev Polynomial Basis of Degree 9
    # The domain is [-3, 3]. Chebyshev polynomials are orthogonal on [-1, 1].
    # We map x in [-3, 3] to t in [-1, 1] using t = x / 3.
    # The model is: y = sum_{k=0}^{9} p[k] * T_k(x/3)
    # This uses all 10 parameters as coefficients for the 10 Chebyshev polynomials.
    # Chebyshev polynomials provide a stable, orthogonal basis for function approximation,
    # allowing the model to fit smooth non-linear functions much better than a linear seed,
    # while avoiding numerical instability of raw high-degree monomials.
    
    # Map x to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    # We can compute these iteratively.
    # T0 = ones_like(t)
    # T1 = t
    # T2 = 2*t*T1 - T0
    # ...
    
    # Let's build a matrix of Chebyshev values or just sum them directly.
    # Direct summation is efficient enough for 200 points.
    
    T_prev = np.ones_like(t) # T_0
    T_curr = t.copy()        # T_1
    
    result = p[0] * T_prev
    if len(p) > 1:
        result = result + p[1] * T_curr
    
    for k in range(2, 10):
        # T_k = 2*t*T_{k-1} - T_{k-2}
        T_next = 2.0 * t * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
        if k < len(p):
            result = result + p[k] * T_curr
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to a simple linear combination of first two params
        # Note: In Chebyshev basis, p[0] is constant, p[1] is linear.
        return p[0] + p[1] * x / 3.0
    
    return result
# EVOLVE-BLOCK-END
