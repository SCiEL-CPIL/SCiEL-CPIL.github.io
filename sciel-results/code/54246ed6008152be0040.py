import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    timeout = 290.0

    # Attempt linear time-invariant solution
    # 1. Extract A
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() - start_time > timeout:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # f expects (t, y)
        A[:, j] = f(0.0, e_j)
        
    if time.time() - start_time > timeout:
        # Fallback
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]

    # 2. Verify linearity and time-invariance
    # Use a deterministic test vector to avoid randomness issues if f is chaotic? 
    # No, linear systems don't care.
    # Use a vector with non-zero entries to be robust.
    y_test = np.ones(n)
    t_test = 0.5 * dt
    
    y_expected = A @ y_test
    y_actual = f(t_test, y_test)
    
    # Check relative error
    diff = np.linalg.norm(y_actual - y_expected)
    scale = np.linalg.norm(y_expected)
    
    if diff < 1e-12 * (1.0 + scale):
        # 3. Solve using matrix exponential
        M = A * dt
        try:
            expM = scipy.linalg.expm(M)
            y_end = expM @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass # Fall through to solver if expm fails
            
    # Fallback: High accuracy adaptive solver
    # DOP853 is good for non-stiff. 
    # If stiff, it might struggle, but n=100 is small.
    # Radau is better for stiff but expensive in calls if Jacobian is finite-diff.
    # We will try DOP853 first.
    sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]
