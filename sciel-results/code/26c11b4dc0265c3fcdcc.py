"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

# Robertson problem constants
K1 = 0.04
K2 = 3.0e-7
K3 = 1.0e4

def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson problem."""
    # y[0], y[1], y[2]
    # y1' = -k1*y1 + k3*y2*y3
    # y2' = k1*y1 - k2*y2^2 - k3*y2*y3
    # y3' = k2*y2^2
    return np.array([
        [-K1, K3 * y[2], K3 * y[1]],
        [K1, -2.0 * K2 * y[1] - K3 * y[2], -K3 * y[1]],
        [0.0, 2.0 * K2 * y[1], 0.0]
    ])

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit, keep margin
    
    # We assume the problem is Robertson based on the task description.
    # If f is not Robertson, we might not have the correct Jacobian.
    # However, the task says "fixed Robertson initial-value problem".
    # To be safe, we can check if f matches Robertson structure by calling it once?
    # No, that wastes calls. The task contract says "fixed Robertson".
    # We will assume f is the Robertson RHS.
    
    # Strategy: Two-stage integration for higher precision at the end.
    # Stage 1: t0 to t_mid
    # Stage 2: t_mid to t1
    
    # Choose a midpoint. The knowledge suggests t~39.
    # Let's pick t_mid = 39.0 if t1 is 40.0.
    # If t_span is different, we scale it.
    total_span = t1 - t0
    t_mid = t0 + 0.975 * total_span  # ~39 for [0, 40]
    
    y_current = y0.copy()
    
    # Stage 1
    sol1 = scipy.integrate.solve_ivp(
        f, 
        (t0, t_mid), 
        y_current, 
        method='Radau', 
        jac=_robertson_jac,
        rtol=1e-13, 
        atol=1e-15,
        max_step=total_span / 100.0 # Prevent huge steps, though Radau adapts
    )
    
    if not sol1.success:
        # Fallback to single stage with default tight tolerances
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=1e-12, 
            atol=1e-14
        )
        return sol.y[:, -1]
        
    y_current = sol1.y[:, -1]
    
    # Check time
    if time.time() > deadline - 10.0:
        # Not enough time for stage 2, return current best (at t_mid, which is close to t1)
        # Actually, we need y(t1). If we can't finish, we should probably just run a single pass 
        # with looser tolerances that fits in time? 
        # But Radau is fast. Let's assume it fits.
        # If we are close to deadline, we return the result of a single pass with moderate tolerances.
        # But wait, we already spent time on stage 1.
        # Let's just try stage 2. If it fails due to time, we are in trouble.
        # Given the speed of Radau on this problem, it should be < 1s.
        pass

    # Stage 2: t_mid to t1
    sol2 = scipy.integrate.solve_ivp(
        f, 
        (t_mid, t1), 
        y_current, 
        method='Radau', 
        jac=_robertson_jac,
        rtol=1e-14, 
        atol=1e-16
    )
    
    if not sol2.success:
        # Fallback if stage 2 failed
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='Radau', 
            jac=_robertson_jac,
            rtol=1e-12, 
            atol=1e-14
        )
        return sol.y[:, -1]

    return sol2.y[:, -1]

# EVOLVE-BLOCK-END
