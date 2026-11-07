"""Symbolic regression: recover hidden univariate f on [-3,3] with 10 free params.

DIRECT approach: use a rich, fixed, smooth, bounded basis of 10 functions of x.
The model is strictly LINEAR in the 10 params:
    f(x) = sum_{i=0}^{9} p_i * phi_i(x)
Because the model is linear in params, the training MSE is a convex quadratic in
params with no local minima and no non-differentiable kinks, so BFGS (started at
ones) converges to the least-squares solution in a handful of iterations. This
removes the amplitude/frequency coupling and abs() kinks of the seed model that
caused the enormous score.

The 10 basis functions span even/odd polynomials (deg <= 3), first and second
harmonics, a saturating odd function, and a rational even bump -- a diverse set
that approximates a wide class of univariate targets on [-3,3]. All basis values
are bounded on [-3,3], so no overflow and smooth numeric gradients.

Per-call cost is O(n) vectorized numpy (a few tens of microseconds for n=200),
far below the 150 s wall-clock budget; BFGS issues many such calls, so we keep
the body allocation-light and branch-free. No internal search loop exists, so an
active deadline guard would only add per-call overhead without interrupting any
work; the function simply returns the closed-form prediction.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Use Chebyshev Polynomials of the First Kind.
    
    The previous model used an ad-hoc mix of polynomials, trigonometric functions,
    and saturating functions. This model uses the first 10 Chebyshev polynomials 
    T_k(t) on the mapped interval t in [-1, 1].
    
    Chebyshev polynomials are orthogonal and provide superior approximation 
    properties (minimax error) for smooth functions on a bounded interval.
    The model remains linear in parameters: y = sum p_k * T_k(t).
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    # t = (x - 0) / 3
    t = x / 3.0
    
    # Construct Chebyshev basis T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    phi = np.empty((10, x.size), dtype=np.float64)
    phi[0] = 1.0
    phi[1] = t
    
    if x.size > 0:
        # Recurrence for k >= 2
        t_prev2 = phi[0] # T_0
        t_prev1 = phi[1] # T_1
        for k in range(2, 10):
            t_curr = 2.0 * t * t_prev1 - t_prev2
            phi[k] = t_curr
            t_prev2 = t_prev1
            t_prev1 = t_curr

    # Linear combination: y = p @ phi
    y = p @ phi

    # Safety: guarantee finiteness
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    return y
# EVOLVE-BLOCK-END
