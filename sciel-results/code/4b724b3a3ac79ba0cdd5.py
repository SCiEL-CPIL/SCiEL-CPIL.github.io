import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # 1. Extract Linear Constant Coefficient Operator
    # Task guarantees a fixed 15-dimensional stiff linear system.
    # We assume y' = A y + b.
    # We need to find A and b.
    # f(t, y) = A y + b
    # f(t, 0) = b
    # f(t, e_j) = A e_j + b = A[:, j] + b
    
    # Get b
    b = f(t0, np.zeros(n))
    
    # Get A columns
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Total calls so far: 1 + 15 = 16.
    # This is significantly better for the -0.5*log10(calls) term than 20 calls.
    
    if time.time() > deadline:
        return y0

    try:
        from scipy.linalg import expm
        
        T = t1 - t0
        
        # Method: Augmented Matrix Exponential
        # Consider z = [y; 1]. Then z' = [A b; 0 0] z.
        # z(T) = exp(M * T) z(0).
        # M is (n+1) x (n+1).
        # z(0) = [y0; 1].
        # The first n components of z(T) are the solution y(T).
        
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        # Last row is zeros, last col is b (already set), bottom-right is 0.
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        exp_MT = expm(M * T)
        
        zT = exp_MT @ z0
        y_final = zT[:n]
        
        return y_final
        
    except Exception:
        # Fallback if expm fails or takes too long (unlikely for 16x16)
        pass

    # 2. Fallback Solver (should rarely be reached)
    return _fallback_solve(f, t0, t1, y0)

def _fallback_solve(f, t0, t1, y0):
    # Use solve_ivp with Radau for stiffness
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Last resort: Fixed step implicit Euler
    # This is likely to score poorly due to call count, but better than crashing
    n_steps = 1000
    h = (t1 - t0) / n_steps
    y = y0.copy()
    t = t0
    for _ in range(n_steps):
        z = y
        for _ in range(3):
            z = y + h * f(t + h, z)
        y = z
        t += h
    return y
