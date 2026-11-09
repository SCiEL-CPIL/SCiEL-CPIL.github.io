"""ODE solver: 15-dimensional stiff linear system - minimal call count approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """Integrate a 15-dim stiff linear IVP using minimal function calls and matrix exponential."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    try:
        # Strategy: Assume homogeneous linear system y' = A*y (most common for stiff linear benchmarks)
        # Extract A using n calls total:
        #   Call 1: f(t0, y0) gives A @ y0
        #   Calls 2..n: f(t0, e_i) for i=1..n-1 gives columns 1..n-1 of A
        #   Column 0: derived from A @ y0 and known columns if y0[0] != 0
        #   If y0[0] == 0, use f(t0, e_0) as call n+1 (but we only need n if y0 is generic)
        
        # More robust: Use n calls to get all columns directly
        # f(t0, e_i) = A @ e_i = column i of A (for homogeneous system)
        # But this requires n calls, and we need to verify homogeneity.
        
        # Actually, for a homogeneous system y' = A*y:
        # f(t, y) = A*y for all t, y
        # So f(t0, e_i) = column i of A, for i = 0, ..., n-1
        # Total: n calls to extract A
        
        # But we also need to verify the system is homogeneous (f(t, 0) = 0)
        # and time-invariant. Let's use a smart approach:
        
        # Call 1: Check f(t0, 0) to see if b is zero
        f_t0_zero = f(t0, np.zeros(n))
        b_is_zero = np.allclose(f_t0_zero, 0.0, atol=1e-12)
        
        if not b_is_zero:
            # Inhomogeneous: use n+1 calls (b + n columns of A)
            b = f_t0_zero
            A = np.zeros((n, n))
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                A[:, i] = f(t0, e_i) - b
            # Total calls: 1 + n = n+1
            
            # Verify time-invariance with 2 more calls (b at t1, one column at t1)
            b1 = f(t1, np.zeros(n))
            e_0 = np.zeros(n)
            e_0[0] = 1.0
            A_col0_t1 = f(t1, e_0) - b1
            
            is_time_invariant = np.allclose(b, b1, rtol=1e-8, atol=1e-8) and \
                                np.allclose(A[:, 0], A_col0_t1, rtol=1e-8, atol=1e-8)
            
            if is_time_invariant:
                # Solve using augmented matrix
                M = np.zeros((n + 1, n + 1))
                M[:n, :n] = A
                M[:n, n] = b
                z0 = np.zeros(n + 1)
                z0[:n] = y0
                z0[n] = 1.0
                T = t1 - t0
                exp_M_T = expm(M * T)
                z1 = exp_M_T @ z0
                return z1[:n]
        else:
            # Homogeneous: y' = A*y
            # Extract A using n calls: f(t0, e_i) = column i of A
            A = np.zeros((n, n))
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                A[:, i] = f(t0, e_i)
            # Total calls: 1 (for zero check) + n = n+1
            # But wait, we already made 1 call for f_t0_zero, then n calls for columns
            # That's n+1 calls total.
            
            # Verify time-invariance: check one column at t1
            e_0 = np.zeros(n)
            e_0[0] = 1.0
            A_col0_t1 = f(t1, e_0)
            
            is_time_invariant = np.allclose(A[:, 0], A_col0_t1, rtol=1e-8, atol=1e-8)
            
            if is_time_invariant:
                T = t1 - t0
                exp_AT = expm(A * T)
                return exp_AT @ y0
                
    except Exception:
        pass
    
    # Fallback: Use solve_ivp with Radau
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
    
    # Final fallback: RK4
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
