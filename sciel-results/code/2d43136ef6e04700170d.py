import numpy as np
import time

# Hard deadline guard: return safely before the 150 s limit.
_START = time.time()
_DEADLINE = _START + 140.0

def _chebyshev_basis(x: np.ndarray) -> np.ndarray:
    """
    Build a Chebyshev polynomial basis of degree 0..9 on the interval [0.1, 6].
    We map x from [0.1, 6] to t in [-1, 1] via t = (2x - (x_min + x_max)) / (x_max - x_min).
    Then compute T_k(t) for k=0..9.
    T_0(t) = 1
    T_1(t) = t
    T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    This gives near-orthogonal basis functions, improving conditioning over monomials.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    x_min = 0.1
    x_max = 6.0
    # Map to [-1, 1]
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    
    N = x.size
    B = np.empty((N, 10), dtype=np.float64)
    
    # T_0
    B[:, 0] = 1.0
    if N == 0:
        return B
    
    # T_1
    B[:, 1] = t
    
    # T_k for k >= 2
    for k in range(2, 10):
        B[:, k] = 2.0 * t * B[:, k-1] - B[:, k-2]
    
    return B

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form predictor using Chebyshev polynomial basis.
    Linear in parameters, so BFGS converges quickly to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    x = x.astype(np.float64)
    
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(x)
    
    B = _chebyshev_basis(x)
    
    out = B @ p
    if not np.all(np.isfinite(out)):
        return np.zeros_like(x)
    return out
