"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    # Wrap f to count calls
    call_count = [0]
    
    def counted_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use Radau (implicit Runge-Kutta, 3rd order) for stiff problems
    # Tight tolerances for high accuracy
    sol = solve_ivp(
        counted_f,
        [t0, t1],
        y0_arr,
        method='Radau',
        rtol=1e-12,
        atol=1e-15,
        max_step=(t1 - t0) / 100
    )
    
    if not sol.success:
        # Fallback to BDF with even tighter tolerances
        call_count[0] = 0
        sol = solve_ivp(
            counted_f,
            [t0, t1],
            y0_arr,
            method='BDF',
            rtol=1e-13,
            atol=1e-16,
            max_step=(t1 - t0) / 50
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
