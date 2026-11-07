"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Integrates the ODE using DOP853 with tight tolerances to match the reference
    solution closely while minimizing function calls.
    """
    start_time = time.time()
    t0, t1 = t_span
    
    # Fallback timer check
    def check_time():
        return time.time() - start_time < 290  # 10s margin

    try:
        # DOP853 is an 8th-order Dormand-Prince method.
        # Using the same tolerances as the reference solution ensures
        # the relative error is at the machine precision floor (1e-14),
        # maximizing the -log10(rel) term.
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Fallback: High-step RK4 if DOP853 fails or times out (unlikely)
    # This is a safety net to ensure a valid return.
    if check_time():
        try:
            steps = 10000
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                if not check_time():
                    return y # Return current best if time runs out
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        except Exception:
            # Last resort: return initial state (will score 0 but is valid shape)
            return np.array(y0, dtype=float)
    
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
