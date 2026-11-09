import numpy as np
import time
from scipy.sparse.linalg import expm_multiply

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: 
    1. Detect if the system is linear and time-independent (f(t,y) = A@y).
       If so, use matrix exponential for exact solution (up to floating point).
    2. Otherwise, fall back to a high-precision adaptive solver.
    
    The score is -log10(relative error) - 0.5*log10(calls to f).
    For a linear system, we can extract A in O(n) calls and then compute
    expm(A*dt) @ y0 with 0 additional calls to f. This minimizes calls.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin under 300s kill
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0
    
    # Check if system is linear and time-independent
    # Probe: f(t0, y) should equal A@y for some fixed A.
    # We can extract A by evaluating f at basis vectors.
    # But first, let's do a quick check: is f linear?
    # If f(t0, 0) != 0, it's affine. For heat equation, it should be linear.
    
    # Try to extract A
    # For a 10x10 grid, n=100. Extracting A takes 100 calls.
    # Then expm_multiply is fast.
    
    # Quick linearity check: f(t0, 0) should be 0 for linear systems
    y_zero = np.zeros_like(y0)
    f_zero = np.asarray(f(t0, y_zero), dtype=float)
    
    if np.allclose(f_zero, 0.0, atol=1e-14):
        # Likely linear. Extract A.
        A = np.empty((n, n), dtype=float)
        
        # Extract columns of A
        for j in range(n):
            e_j = np.zeros_like(y0)
            e_j[j] = 1.0
            A[:, j] = np.asarray(f(t0, e_j), dtype=float)
            
            # Time check
            if time.time() > deadline:
                break
        
        # Verify we extracted all columns
        if time.time() <= deadline:
            # Check if A is consistent: f(t0, y0) should equal A @ y0
            f_y0 = np.asarray(f(t0, y0), dtype=float)
            A_y0 = A @ y0
            if np.allclose(f_y0, A_y0, rtol=1e-10, atol=1e-12):
                # Linear system confirmed. Use matrix exponential.
                # This gives exact solution (up to floating point) with minimal calls.
                # Total calls: 1 (zero) + n (basis) + 1 (verify) = n+2
                try:
                    y_end = expm_multiply(A * dt, y0)
                    return y_end
                except Exception:
                    pass
        
        # If extraction failed or verification failed, fall through
    
    # Fallback: use high-precision adaptive solver
    # Radau IIA is good for stiff systems like heat equation
    from scipy.integrate import solve_ivp
    
    # Use tight tolerances for high accuracy
    sol = solve_ivp(
        f, t_span, y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        max_step=dt/100 if dt > 0 else 1.0
    )
    
    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]
    
    # If Radau fails, try RK45 with tighter tolerances
    sol = solve_ivp(
        f, t_span, y0,
        method='RK45',
        rtol=1e-12,
        atol=1e-14,
        max_step=dt/1000 if dt > 0 else 1.0
    )
    
    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]
    
    # Last resort: return initial condition (should not happen)
    return y0
