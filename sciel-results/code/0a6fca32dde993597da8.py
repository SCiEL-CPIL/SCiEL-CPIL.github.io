"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the 10x10 2D Heat Equation.
    Strategy:
    1. Infer the system matrix A directly by probing f with basis vectors.
       This handles any linear, autonomous ODE (y' = A y) exactly,
       regardless of the specific discretization details (BCs, grid size).
    2. Compute the exact solution using the matrix exponential expm(A * dt).
    3. Fallback to a robust implicit solver if linearity is not detected or expm fails.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # We assume the system is linear and autonomous: y' = A y.
    # We can determine A by evaluating f(t, e_i) for each basis vector e_i.
    # This requires n evaluations. For n=100, this is very cheap (100 calls).
    # The total call budget is 2,000,000, so 100 is negligible.
    
    try:
        A = np.zeros((n, n))
        # Probe each column of A
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # f(t, y) = A @ y. So f(t, e_i) = A @ e_i = A[:, i]
            A[:, i] = np.array(f(t0, e_i))
            
            # Time check (should never trigger for 100 calls)
            if time.time() > deadline:
                raise TimeoutError()
        
        # Check if the system is indeed linear by verifying f(t, y0) == A @ y0
        # This is a good sanity check.
        f_y0 = np.array(f(t0, y0))
        A_y0 = A @ y0
        residual = np.linalg.norm(f_y0 - A_y0)
        norm_A_y0 = np.linalg.norm(A_y0)
        
        # If the residual is small, we trust A.
        # For a linear system, this should be ~1e-16.
        if residual < 1e-8 * (norm_A_y0 + 1e-10):
            dt = t1 - t0
            # Compute exact solution
            E = expm(A * dt)
            y_end = E @ y0
            
            if np.all(np.isfinite(y_end)):
                return y_end
        else:
            # If not linear, or probe failed, fallback
            pass
            
    except Exception:
        pass
    
    # Fallback: Robust implicit solver
    return _fallback_solve(f, t_span, y0, start_time)

def _fallback_solve(f, t_span, y0, start_time):
    """
    Robust fallback using SciPy's Radau method (implicit, stiff-accurate).
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set a tight tolerance to maximize accuracy score.
    # rtol=1e-10, atol=1e-12
    # Radau is good for stiff problems like heat equation.
    
    # We need to be careful with the number of calls.
    # Radau might use many Jacobian evaluations if not provided.
    # But we don't have the Jacobian analytically here (or we do, but the probe failed).
    # Let's just let it estimate.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1-t0)/10 # Limit step size to ensure accuracy
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: High-order RK4 with many steps
    # This is slow and less accurate for stiff systems, but better than failing.
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        if time.time() > start_time + 300:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
