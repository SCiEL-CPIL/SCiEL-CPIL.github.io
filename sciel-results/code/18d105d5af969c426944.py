"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) on [t0, t1] with initial condition y0.
    Assumes the system is linear and autonomous (y' = A y + b) based on the 
    convection-diffusion problem description.
    Uses the augmented matrix exponential method for exact solution of the linear system.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    if dt == 0:
        return y0

    # 1. Extract the system matrix A and bias b.
    # f(t, y) = A y + b
    # b = f(t, 0)
    # A[:, j] = f(t, e_j) - b
    
    try:
        b = np.array(f(t0, np.zeros(n)), dtype=float)
    except Exception:
        b = np.zeros(n)

    A = np.zeros((n, n))
    for j in range(n):
        if time.time() > deadline:
            # If we run out of time during extraction, fall back
            return _fallback_solve(f, t_span, y0, start_time, deadline)
        
        e_j = np.zeros(n)
        e_j[j] = 1.0
        try:
            val = np.array(f(t0, e_j), dtype=float)
            A[:, j] = val - b
        except Exception:
            # If extraction fails for a column, we can't reliably use Matrix Exp
            return _fallback_solve(f, t_span, y0, start_time, deadline)

    # Check for finite values
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return _fallback_solve(f, t_span, y0, start_time, deadline)

    # 2. Compute the solution using the augmented matrix exponential.
    # System: z' = M z, where z = [y; 1]
    # M = [[A, b], [0, 0]]
    # z(t) = expm(M * dt) @ z0
    
    try:
        # Create augmented matrix
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        # M[n, :] is already 0
        
        # Initial state vector
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        # Compute matrix exponential
        # scipy.linalg.expm is robust for small/medium matrices
        E = scipy.linalg.expm(M * dt)
        
        # Final state
        z_end = E @ z0
        y_end = z_end[:n]
        
        # Verify result is finite
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    # Fallback if Matrix Exponential fails or produces non-finite results
    return _fallback_solve(f, t_span, y0, start_time, deadline)

def _fallback_solve(f, t_span, y0, start_time, deadline):
    """
    Fallback solver using scipy.integrate.solve_ivp.
    Uses Radau method for stiff problems (common in diffusion) with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try Radau first as it handles stiffness well
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12, 
            max_step=(t1-t0)/100 # Ensure enough steps for accuracy if adaptive takes large steps
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # If Radau fails, try DOP853 (high order explicit)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: Fixed step RK4 if all else fails
    # This is a safety net to ensure we return *something* rather than crashing
    y = np.array(y0, dtype=float)
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y

# EVOLVE-BLOCK-END
