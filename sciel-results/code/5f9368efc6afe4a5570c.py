import numpy as np
import scipy.linalg
import time
import sys

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    # 1. Check if the system is linear and time-independent
    # Heat equation is linear: f(t, y) = A @ y
    # We verify linearity with a few probes.
    
    # Probe 0: f(0, 0) should be 0 for homogeneous heat equation
    # But let's just check linearity.
    
    # Generate two random vectors
    np.random.seed(42)
    v1 = np.random.randn(n)
    v2 = np.random.randn(n)
    
    f_v1 = f(t0, v1)
    f_v2 = f(t0, v2)
    f_v12 = f(t0, v1 + v2)
    
    # Check linearity: f(v1 + v2) == f(v1) + f(v2)
    # Also check time independence: f(t1, v1) == f(t0, v1)
    f_v1_t1 = f(t1, v1)
    
    is_linear = np.allclose(f_v12, f_v1 + f_v2, rtol=1e-10, atol=1e-12)
    is_time_independent = np.allclose(f_v1_t1, f_v1, rtol=1e-10, atol=1e-12)
    
    if is_linear and is_time_independent:
        # 2. Recover the matrix A
        # A @ v = f(t, v)
        # We need A. We can solve for A by using basis vectors.
        # A = F @ E^{-1} where F is the matrix of f(e_i) and E is identity.
        # So A = F.
        
        A = np.empty((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f(t0, e_i)
            
        if time.time() > deadline:
             # Fallback if something went wrong or is slow, though expm is fast
             return _fallback_solve(f, t_span, y0)
        
        # 3. Compute matrix exponential
        # y(t1) = exp(A * dt) @ y0
        M = scipy.linalg.expm(A * dt)
        
        # 4. Apply to initial state
        y_end = M @ y0
        
        return y_end
    else:
        # Fallback to high-precision ODE solver
        return _fallback_solve(f, t_span, y0)

def _fallback_solve(f, t_span, y0):
    """
    Fallback for non-linear or time-dependent systems.
    Uses scipy.integrate.solve_ivp with high precision.
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # DOP853 is a high-order explicit method, good for non-stiff problems.
    # Radau is implicit, good for stiff problems.
    # Heat equation is stiff, but if it's non-linear, it might be less stiff or we need implicit.
    # Let's try DOP853 first with tight tolerances.
    
    sol = solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1-t0)/100 # Limit step size to ensure accuracy
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # If DOP853 fails, try Radau (implicit)
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        return sol.y[:, -1]
