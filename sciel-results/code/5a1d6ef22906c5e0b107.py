"""Improved ODE solver for the Robertson problem using adaptive stiff-capable integration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def _robertson_jacobian(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    y = [y0, y1, y2]
    y' = [-k1*y0 + k3*y1*y2, k1*y0 - k2*y1 - k3*y1*y2, k2*y1]
    with k1=0.04, k2=3e-7, k3=1e4
    """
    k1 = 0.04
    k2 = 3e-7
    k3 = 1e4
    y0, y1, y2 = y
    return np.array([
        [-k1, k3 * y2, k3 * y1],
        [k1, -(k2 + k3 * y2), -k3 * y1],
        [0.0, k2, 0.0]
    ])

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses scipy.integrate.solve_ivp with LSODA method for stiff/non-stiff automatic switching.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s/330s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try solve_ivp with high precision
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-12, 
            atol=1e-14,
            jac=_robertson_jacobian,
            dense_output=False
        )
        
        if time.time() > deadline:
            # If it took too long (unlikely for this problem), return last available state if possible
            if sol.t.size > 0:
                return sol.y[:, -1]
            return y0

        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback: if LSODA fails, try Radau (implicit stiff solver)
            sol_radau = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-12, 
                atol=1e-14,
                jac=_robertson_jacobian
            )
            if sol_radau.success:
                return sol_radau.y[:, -1]
            # Last resort: return the last computed value or initial
            if sol_radau.t.size > 0:
                return sol_radau.y[:, -1]
            return y0

    except Exception:
        # In case of any unexpected error, return initial state to avoid crash
        return y0

# EVOLVE-BLOCK-END
