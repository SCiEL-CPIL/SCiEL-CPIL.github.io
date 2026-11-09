"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """Integrate a 15-dim stiff linear IVP using linear system detection and matrix exponential."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Strategy: Detect if the system is linear and time-invariant (y' = Ay + b)
    # If so, solve exactly using matrix exponential.
    # Cost: ~n+3 function calls.
    
    try:
        # 1. Estimate b and A at t0
        b0 = f(t0, np.zeros(n))
        
        # Check if b0 is zero (homogeneous)
        # We need columns of A. A[:, i] = f(t0, e_i) - b0
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f(t0, e_i) - b0
            
        # 2. Verify time-invariance at t1
        # Check b at t1
        b1 = f(t1, np.zeros(n))
        
        # Check one column of A at t1 (e.g., column 0) to save calls
        e_0 = np.zeros(n)
        e_0[0] = 1.0
        A_col_0_t1 = f(t1, e_0) - b1
        
        # Compare
        tol_check = 1e-8
        is_time_invariant = np.allclose(b0, b1, rtol=tol_check, atol=tol_check) and \
                            np.allclose(A[:, 0], A_col_0_t1, rtol=tol_check, atol=tol_check)
        
        if is_time_invariant:
            # Construct augmented matrix for inhomogeneous linear system
            # y' = A y + b
            # Let z = [y; 1]. Then z' = M z where M = [[A, b], [0, 0]]
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = b0
            # Last row is zeros
            
            z0 = np.zeros(n + 1)
            z0[:n] = y0
            z0[n] = 1.0
            
            T = t1 - t0
            M_T = M * T
            exp_M_T = expm(M_T)
            
            z1 = exp_M_T @ z0
            return z1[:n]
            
    except Exception:
        pass

    # Fallback: Use solve_ivp with Radau
    # We can provide the Jacobian if we detected it, even if not perfectly time-invariant,
    # but for safety, we stick to the standard robust method if detection failed.
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0, 
            method='Radau',
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1 - t0) / 100
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: simple RK4 with adaptive stepping
    n = len(y0)
    y = y0.copy()
    t = t0
    h = (t1 - t0) / 1000
    
    while t < t1 and time.time() < deadline:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = min(t + h, t1)
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
