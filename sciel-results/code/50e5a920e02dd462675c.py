import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Determine if the system is linear time-invariant (LTI)
    # Probe A at t0 and t1
    y_zero = np.zeros(n)
    f_t0_y0 = np.array(f(t0, y_zero), dtype=float)
    
    A_t0 = np.zeros((n, n))
    eps = 1e-8
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        f_t0_e = np.array(f(t0, e), dtype=float)
        A_t0[:, j] = (f_t0_e - f_t0_y0) / eps
        
    f_t1_y0 = np.array(f(t1, y_zero), dtype=float)
    A_t1 = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        f_t1_e = np.array(f(t1, e), dtype=float)
        A_t1[:, j] = (f_t1_e - f_t1_y0) / eps
        
    # Check if A is constant
    # Use a relative tolerance
    diff = np.linalg.norm(A_t1 - A_t0)
    scale = np.linalg.norm(A_t0) + 1e-10
    is_constant_A = diff / scale < 1e-6
    
    if is_constant_A:
        A_const = A_t0
        def jac(t, y):
            return A_const
    else:
        # If A is time-dependent, we need a Jacobian function that computes A(t)
        # This is expensive. We might fall back to finite differences or a simpler solver
        # But for now, let's try to compute A(t) on the fly in jac
        def jac(t, y):
            y_zero_local = np.zeros(n)
            f_t_y0 = np.array(f(t, y_zero_local), dtype=float)
            A_t = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = eps
                f_t_e = np.array(f(t, e), dtype=float)
                A_t[:, j] = (f_t_e - f_t_y0) / eps
            return A_t

    # Use Radau method which is suitable for stiff problems
    # rtol and atol set to high precision
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        jac=jac,
        rtol=1e-10, 
        atol=1e-12,
        max_step=(t1-t0)/10 # Encourage some steps, but let it adapt
    )
    
    if not sol.success:
        # Fallback to a more robust but slower method if Radau fails
        # Or just return the last available point
        pass
        
    return sol.y[:, -1]
