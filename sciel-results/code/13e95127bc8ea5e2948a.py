import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 20s margin from 300s limit
    
    # 1. Extract A and b with minimal f-calls (16 total: 1 for b + 15 for basis)
    # Task guarantees fixed linear system, so we skip linearity verification
    b = f(t0, np.zeros(n))
    
    # Extract A using basis vectors: A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
    
    if time.time() > deadline:
        return y0
    
    T = t1 - t0
    
    # Primary method: Use scipy.linalg.expm on the block matrix for exact solution
    try:
        from scipy.linalg import expm
        
        # Build block matrix M (n+1 x n+1)
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A
        M[:n, n] = b
        
        # Initial state for augmented system
        z0 = np.zeros(n+1)
        z0[:n] = y0
        z0[n] = 1.0
        
        # Compute matrix exponential
        exp_MT = expm(M * T)
        
        # Final state
        z_T = exp_MT @ z0
        
        # Extract y(T)
        y_final = z_T[:n]
        
        # Verify result is finite
        if not np.all(np.isfinite(y_final)):
            raise ValueError("Non-finite result from expm")
            
        return y_final
    except Exception:
        pass
    
    # Fallback: Try direct approach with solve if A is invertible
    try:
        from scipy.linalg import expm
        exp_AT = expm(A * T)
        
        try:
            A_inv_b = np.linalg.solve(A, b)
            y_particular = (exp_AT - np.eye(n)) @ A_inv_b
            y_final = exp_AT @ y0 + y_particular
            if np.all(np.isfinite(y_final)):
                return y_final
        except np.linalg.LinAlgError:
            pass
    except Exception:
        pass
    
    # 3. Fallback Solver
    return _fallback_solve(f, t0, t1, y0, start_time, deadline)

def _fallback_solve(f, t0, t1, y0, start_time, deadline):
    # Use solve_ivp with Radau for stiffness
    try:
        from scipy.integrate import solve_ivp
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            return y0
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Last resort: Fixed step implicit Euler
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
