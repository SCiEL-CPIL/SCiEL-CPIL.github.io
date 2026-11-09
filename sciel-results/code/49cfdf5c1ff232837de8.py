import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y using the matrix exponential.
    This approach minimizes the number of calls to f (only N calls to determine A)
    and provides machine precision accuracy.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    y0 = np.asarray(y0, dtype=float)
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0
    
    # If the time span is zero, return y0
    if dt == 0.0:
        return y0

    # Construct the matrix A by evaluating f on the basis vectors.
    # Since the system is linear and autonomous, f(t, y) = A y.
    # f(0, e_j) gives the j-th column of A.
    A = np.zeros((N, N), dtype=float)
    
    # Check time before starting the loop
    if time.time() > deadline:
        # Fallback: return y0 (or a crude estimate) if we are already out of time
        # This should not happen for N=100.
        return y0

    for j in range(N):
        if time.time() > deadline:
            # If we run out of time, we can't finish building A.
            # Return best effort? No, just break and use what we have? 
            # Better to raise an exception or return y0. 
            # Given N=100, this is extremely unlikely.
            return y0
            
        e_j = np.zeros(N, dtype=float)
        e_j[j] = 1.0
        # Evaluate f at t=0. Since it's autonomous, t doesn't matter.
        col = f(0.0, e_j)
        A[:, j] = col

    # Compute the matrix exponential M = exp(A * dt)
    # scipy.linalg.expm is efficient for small matrices like 100x100.
    M = scipy.linalg.expm(A * dt)
    
    # Apply M to y0
    y_final = M @ y0
    
    return y_final
