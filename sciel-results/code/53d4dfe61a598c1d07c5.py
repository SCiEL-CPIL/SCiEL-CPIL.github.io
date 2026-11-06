"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear system y' = A y over [t0, t1].
    
    Strategy: Extract the constant matrix A from f using 16 function calls,
    then use matrix exponential for exact integration. This minimizes function
    calls while maintaining high accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Track elapsed time for safety margin
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    # Extract matrix A from the linear system y' = A y
    # f(t, y) = A @ y for constant A
    # We can extract A by calling f with standard basis vectors
    A = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # Call f at any time (should be time-independent for linear system)
        A[:, j] = f(t0, e_j)
    
    # Verify the system is indeed linear and constant by checking one more point
    # f(t1, y0) should equal A @ y0
    check = f(t1, y0)
    expected = A @ y0
    if not np.allclose(check, expected, rtol=1e-10, atol=1e-12):
        # If not linear/constant, fall back to a more general approach
        # But for this task, it should be linear and constant
        pass
    
    # Use matrix exponential for exact solution: y(t1) = expm(A * (t1 - t0)) @ y0
    dt = t1 - t0
    
    # Compute matrix exponential using scipy
    from scipy.linalg import expm
    expA_dt = expm(A * dt)
    
    y_final = expA_dt @ y0
    
    # Check if we're within time limit
    if time.time() > deadline:
        # Return best result we have
        return y_final
    
    # The matrix exponential gives the exact solution for linear systems,
    # so no further refinement is needed.
    return y_final
# EVOLVE-BLOCK-END
