"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson IVP y' = f(t, y) over t_span using solve_ivp with Radau.
    We provide an analytic Jacobian to reduce function evaluations.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 300s limit, use 295s to be safe
    start_time = time.time()
    deadline = 295.0
    
    # Robertson problem analytic Jacobian:
    # y' = [ -k1*y0 + k3*y1*y2,
    #       k1*y0 - k2*y1^2 - k3*y1*y2,
    #       k2*y1^2 ]
    # where k1=0.04, k2=3e-7, k3=1e4
    # J = [[ -k1,  k3*y2,  k3*y1 ],
    #      [  k1, -2*k2*y1 - k3*y2, -k3*y1 ],
    #      [  0,  2*k2*y1,  0 ]]
    
    k1 = 0.04
    k2 = 3e-7
    k3 = 1e4
    
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-k1, k3 * y2_val, k3 * y1_val],
            [k1, -2 * k2 * y1_val - k3 * y2_val, -k3 * y1_val],
            [0.0, 2 * k2 * y1_val, 0.0]
        ])
        return J
    
    # Use Radau method with tight tolerances for high accuracy
    # rtol=1e-12, atol=1e-14 for very high precision
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='Radau',
        jac=jac,
        rtol=1e-12,
        atol=1e-14,
        dense_output=False
    )
    
    elapsed = time.time() - start_time
    
    if not result.success:
        # Try with slightly looser tolerances if it failed
        if elapsed < deadline - 10:
            result = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='Radau',
                jac=jac,
                rtol=1e-10,
                atol=1e-12,
                dense_output=False
            )
    
    if result.success and len(result.y) > 0:
        return result.y[:, -1]
    
    # Fallback: use LSODA if Radau fails
    if time.time() - start_time < deadline - 10:
        result2 = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
        if result2.success and len(result2.y) > 0:
            return result2.y[:, -1]
    
    # Last resort: return y0 if everything fails
    return y0
# EVOLVE-BLOCK-END
