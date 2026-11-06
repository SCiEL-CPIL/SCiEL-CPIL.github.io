"""Solve 15-dim stiff linear ODE using high-precision numerical integration."""
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Strategy:
    # 1. Check if the system is time-invariant (constant A).
    #    If so, use expm for high accuracy and low calls.
    # 2. If not, use solve_ivp with tight tolerances.
    
    # Extract A at t0
    def extract_A(t):
        b = np.asarray(f(t, np.zeros(n)), dtype=float)
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            col = np.asarray(f(t, e), dtype=float)
            A[:, j] = col - b
        return A, b

    # Check time invariance
    A0, b0 = extract_A(t0)
    A1, b1 = extract_A(t1)
    
    # If A and b are constant, use expm
    # Tolerance for "constant"
    tol_const = 1e-10
    is_constant = (np.linalg.norm(A0 - A1) < tol_const) and (np.linalg.norm(b0 - b1) < tol_const)
    
    if is_constant:
        # Use expm for constant system
        # Augmented system for affine
        H = np.zeros((n + 1, n + 1), dtype=float)
        H[:n, :n] = A0
        H[:n, n] = b0
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        dt = t1 - t0
        M = expm(dt * H)
        z_end = M @ z0
        return z_end[:n]
    else:
        # Time-varying system: use solve_ivp
        # Radau is good for stiff systems
        # Tight tolerances for high accuracy
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-12,
            max_step=0.01 # Limit step size to ensure accuracy for stiff/fast dynamics
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-10
            )
            
        return sol.y[:, -1]
