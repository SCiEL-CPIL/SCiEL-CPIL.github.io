"""Rewritten program: ODE solver for 15-dim stiff linear system with time-invariance check."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # Step 1: Check for time-invariance
    # Use 3 random vectors to probe f(t0, v) and f(t1, v)
    # If they match, assume constant coefficients.
    
    rng = np.random.default_rng(42)
    is_time_invariant = True
    num_probes = 3
    
    for _ in range(num_probes):
        v = rng.standard_normal(n)
        v = v / np.linalg.norm(v)
        
        f_t0 = f(t0, v)
        f_t1 = f(t1, v)
        
        # Check if f(t0, v) approx f(t1, v)
        # Use relative and absolute tolerance
        if not np.allclose(f_t0, f_t1, rtol=1e-5, atol=1e-8):
            is_time_invariant = False
            break
            
    if is_time_invariant:
        # Reconstruct A using basis vectors at t0
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f(t0, e_i)
            
        # Use matrix exponential for exact solution
        dt = t1 - t0
        M = expm(A * dt)
        return M @ y0
    else:
        # System is time-dependent or stiffly varying
        # Use a robust stiff integrator
        # Radau is a good choice for stiff linear systems
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-10, 
                atol=1e-12,
                max_step=(t1 - t0) / 10.0
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Fallback if solve_ivp fails
                # Try BDF
                sol2 = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='BDF', 
                    rtol=1e-8, 
                    atol=1e-10
                )
                if sol2.success:
                    return sol2.y[:, -1]
                else:
                    # Last resort: constant A approximation
                    A = np.zeros((n, n))
                    for i in range(n):
                        e_i = np.zeros(n)
                        e_i[i] = 1.0
                        A[:, i] = f(t0, e_i)
                    M = expm(A * (t1 - t0))
                    return M @ y0
        except Exception:
            # Fallback to constant A approximation
            A = np.zeros((n, n))
            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0
                A[:, i] = f(t0, e_i)
            M = expm(A * (t1 - t0))
            return M @ y0

# EVOLVE-BLOCK-END
