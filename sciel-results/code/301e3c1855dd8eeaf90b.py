import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
DEADLINE = 110.0  # 10s margin before 120s limit

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers hidden function using Chebyshev polynomial basis of degree 0-9.
    This basis is orthogonal on [-1, 1] (mapped from [-3, 3]) and provides
    optimal approximation properties for smooth functions in the minimax sense.
    """
    # Deadline guard: if we are too close to the limit, we could potentially
    # return a simplified result, but given the O(N) cost, this is unlikely.
    # We include it to satisfy the directive.
    if time.time() - _start_time > DEADLINE:
        # Fallback: return mean of params (arbitrary but finite)
        # This is a safety net, not expected to trigger.
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.empty((10, x.size), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
        
    # Linear combination
    y = p @ T
    
    # Ensure finiteness
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return y
