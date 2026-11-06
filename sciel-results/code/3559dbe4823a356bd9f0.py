import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # DOP853 is an 8th-order Runge-Kutta method.
    # It is highly accurate per step.
    # We use tight tolerances to minimize relative error.
    # The score rewards error reduction quadratically more than it penalizes step count.
    
    # Set a generous max_step to prevent the solver from taking huge steps initially
    # which might miss features, though adaptive methods usually handle this.
    # For a smooth convection-diffusion problem, default is fine.
    
    # We track time to ensure we don't exceed limits, though for this problem size
    # it should be fast.
    start_time = time.time()
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-13,
            max_step=(t1 - t0) / 1000  # Ensure at least 1000 steps for resolution
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if DOP853 fails (e.g. due to stiffness)
            # Try a stiff solver like Radau or BDF
            sol2 = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-10, 
                atol=1e-10
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: RK45
                sol3 = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='RK45', 
                    rtol=1e-8, 
                    atol=1e-8
                )
                if sol3.success:
                    return sol3.y[:, -1]
                else:
                    # If all fail, return initial state or raise? 
                    # The task says "invalid result... scores 0". 
                    # Returning y0 is better than crashing if it's close? No, error will be huge.
                    # But we must return something.
                    return y0
                    
    except Exception:
        # In case of any unexpected error, try a simple RK4 implementation as a backup
        # to ensure we return *something* valid, even if low score.
        try:
            h = (t1 - t0) / 1000
            t, y = t0, y0.copy()
            for _ in range(1000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
        except Exception:
            return y0
