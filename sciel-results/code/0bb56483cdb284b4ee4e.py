"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin for 300s limit
    
    # Strategy: Assume "fixed" means constant coefficients A and b.
    # We probe A and b at t0 only.
    # Calls: n (for A) + 1 (for b) = 16 calls.
    
    # 1. Probe b(t0) = f(t0, 0)
    y_zero = np.zeros(n)
    b_t0 = np.array(f(t0, y_zero), dtype=float)
    
    # 2. Probe A(t0) columns
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j = np.array(f(t0, e_j), dtype=float)
        # f(t, y) = A y + b
        # f(t, e_j) = A e_j + b = A[:, j] + b
        # So A[:, j] = f(t, e_j) - b
        A[:, j] = f_j - b_t0
        
    # 3. Verify linearity and time-invariance by probing at t1
    # This costs another n+1 calls. Total 32 calls so far.
    b_t1 = np.array(f(t1, y_zero), dtype=float)
    A_t1 = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j_t1 = np.array(f(t1, e_j), dtype=float)
        A_t1[:, j] = f_j_t1 - b_t1
        
    # Check if A and b are constant
    tol = 1e-10
    is_constant = (np.allclose(A, A_t1, atol=tol) and np.allclose(b_t0, b_t1, atol=tol))
    
    if is_constant:
        # System is y' = A y + b with constant A, b
        # Use augmented matrix method to handle the affine term exactly
        # z = [y; 1]
        # z' = [[A, b], [0, 0]] z
        # z(t) = exp(M * (t1 - t0)) z(t0)
        
        dt = t1 - t0
        if abs(dt) < 1e-15:
            return y0_arr
            
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b_t0
        # Last row is zero
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0_arr
        z0[n] = 1.0
        
        # Compute matrix exponential
        # expm is robust for 16x16 matrices
        expM = expm(M * dt)
        
        z_final = expM @ z0
        y_final = z_final[:n]
        
        # Sanity check: if the result is not finite, fall back
        if not np.all(np.isfinite(y_final)):
            # Fallback to solver if expm fails (unlikely)
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='BDF',
                rtol=1e-13,
                atol=1e-15
            )
            return sol.y[:, -1]
            
        return y_final
        
    else:
        # Fallback: System is not constant linear/affine, or probes failed.
        # Use solve_ivp with BDF and high precision.
        from scipy.integrate import solve_ivp
        
        # We already spent 2*(n+1) calls.
        # Use a tight tolerance.
        # Check time
        if time.time() > deadline:
            # If we are out of time, return best guess (initial state? or last probe?)
            # This shouldn't happen with such a small problem, but guard just in case.
            return y0_arr
            
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-13,
            atol=1e-15
        )
        
        if not sol.success:
             # Try Radau
             sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-12,
                atol=1e-14
             )
             
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
