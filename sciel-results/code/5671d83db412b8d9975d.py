"""
Static, orthogonal Chebyshev feature basis for symbolic regression.

Strategy:
The evaluator fits 10 parameters using BFGS starting from ones.
Instead of optimizing the basis structure (which is risky and slow),
we use a fixed, universal basis that spans a wide range of functions.
The basis consists of Chebyshev polynomials of the first kind,
mapped from x in [-2, 2] to t in [-1, 1].

Chebyshev polynomials are orthogonal on [-1, 1] with respect to the
weight 1/sqrt(1-t^2), but even without the weight, they provide a
well-conditioned basis for least-squares fitting over [-1, 1].
This avoids the local minima and convergence issues of parametric
basis optimization (Gaussians/Fourier) seen in the previous strategy.

Features:
T_0(t) = 1
T_1(t) = t
T_2(t) = 2t^2 - 1
...
T_9(t)

We use 10 features: T_0 through T_9.
This allows the linear combination to approximate any smooth function
on [-2, 2] reasonably well, and the orthogonality helps BFGS.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # Safety margin

def _chebyshev_t(x):
    """
    Compute Chebyshev polynomials T_0(x) ... T_9(x) for x in [-1, 1].
    Returns array of shape (N, 10).
    """
    N = len(x)
    F = np.zeros((N, 10), dtype=float)
    
    # T_0(x) = 1
    F[:, 0] = 1.0
    
    if N > 0:
        # T_1(x) = x
        F[:, 1] = x
        
        # T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
        for n in range(2, 10):
            F[:, n] = 2.0 * x * F[:, n-1] - F[:, n-2]
            
    return F

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
        
    # Map x from [-2, 2] to t in [-1, 1]
    # t = (x - 0) / 2 = x / 2
    t = x / 2.0
    
    # Compute basis features
    F = _chebyshev_t(t)
    
    # Linear combination
    p = np.asarray(params, dtype=float)
    if p.size != 10:
        # Fallback if params shape is wrong (shouldn't happen)
        p = np.ones(10)
        
    return F @ p
# EVOLVE-BLOCK-END
