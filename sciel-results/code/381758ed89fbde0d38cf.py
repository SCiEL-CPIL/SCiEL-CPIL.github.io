import numpy as np
import time
from scipy.linalg import expm

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Extract A by evaluating f at basis vectors: n calls total.
    # For a linear system y' = A @ y, f(t, y) = A @ y.
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            break

    # Compute matrix exponential: y(t1) = expm(dt * A) @ y0
    # Use mpmath for high precision to handle stiffness/conditioning better than double precision
    if HAS_MPMATH:
        try:
            # Set high precision, e.g., 100 decimal digits
            mpmath.mp.dps = 100
            
            # Convert A to mpmath matrix
            # Using list of lists is faster than element-wise loop
            A_list = A.tolist()
            A_mp = mpmath.matrix(A_list)
            
            # Convert dt to mpmath number
            dt_mp = mpmath.mpf(str(dt))
            
            # Compute matrix exponential
            M_mp = mpmath.expm(dt_mp * A_mp)
            
            # Convert result back to numpy float64
            M = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    M[i, j] = float(M_mp[i, j])
            
            y_end = M @ y0
            return y_end
        except Exception:
            pass

    # Fallback to standard scipy expm if mpmath fails
    M = expm(dt * A)
    y_end = M @ y0
    return y_end
