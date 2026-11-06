"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import odeint


def integrate(f, t_span, y0):
    # Wrap f to count calls
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use odeint (LSODA) which automatically switches between Adams and BDF
    # This is a different approach family than the fixed Radau/BDF in solve_ivp
    # LSODA is often very efficient for stiff problems like Robertson
    try:
        sol = odeint(
            counted_f,
            y0_arr,
            [t0, t1],
            rtol=1e-12,
            atol=1e-15,
            mxstep=100000
        )
        return sol[-1]
    except Exception:
        # Fallback to solve_ivp Radau if odeint fails
        from scipy.integrate import solve_ivp
        call_count[0] = 0
        sol = solve_ivp(
            counted_f,
            [t0, t1],
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-15
        )
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
