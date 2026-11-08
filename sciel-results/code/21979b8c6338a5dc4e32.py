"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    
    # Heuristic: Check if the system is autonomous (independent of t)
    # and potentially linear (constant Jacobian).
    # Convection-diffusion with constant coefficients is autonomous and linear.
    is_autonomous = False
    try:
        f0 = f(t0, y0)
        f1 = f(t1, y0)
        if np.allclose(f0, f1, rtol=1e-5, atol=1e-8):
            is_autonomous = True
    except Exception:
        is_autonomous = False

    jac_func = None
    if is_autonomous:
        # Compute Jacobian at t0, y0 using central differences
        # This assumes the Jacobian is constant (linear system).
        # If the system is non-linear, this is an approximation, but often
        # sufficient for initial steps or if non-linearity is weak.
        J = np.zeros((n, n))
        eps = 1e-7
        for j in range(n):
            yp = y0.copy()
            ym = y0.copy()
            step = eps * max(1.0, abs(y0[j]))
            yp[j] += step
            ym[j] -= step
            try:
                fyp = f(t0, yp)
                fym = f(t0, ym)
                J[:, j] = (fyp - fym) / (2 * step)
            except Exception:
                # If complex step or other fails, fall back to forward diff
                yp = y0.copy()
                step = eps * max(1.0, abs(y0[j]))
                yp[j] += step
                fyp = f(t0, yp)
                J[:, j] = (fyp - f0) / step
                
        def jac_func(t, y):
            return J

    # Try BDF first, which is good for stiff problems
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-10, 
            atol=1e-12,
            jac=jac_func,
            max_step=(t1 - t0) / 5.0
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau if BDF fails
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-10, 
            atol=1e-12,
            jac=jac_func
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to high-order explicit if implicit methods fail
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # If all fail, return initial state (should not happen)
    return y0
# EVOLVE-BLOCK-END
