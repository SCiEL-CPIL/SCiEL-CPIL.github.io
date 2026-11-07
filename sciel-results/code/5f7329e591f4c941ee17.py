"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    # Track time to ensure we finish well before the 300s limit
    start_time = time.time()
    # Use a generous but safe deadline (e.g., 280s)
    deadline = start_time + 280.0
    
    call_count = 0
    
    def f_counted(t, y):
        nonlocal call_count
        # Check deadline
        if time.time() > deadline:
            raise TimeoutError("Exceeded time limit")
        call_count += 1
        return f(t, y)

    try:
        # DOP853 is a high-order (8th order) Runge-Kutta method.
        # It is very accurate and efficient for non-stiff problems like Lorenz.
        # We use tight tolerances to minimize relative error.
        # rtol=1e-13, atol=1e-15 should push the error close to the 1e-14 floor.
        
        sol = scipy.integrate.solve_ivp(
            f_counted,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=1.0  # Allow larger steps, DOP853 will subdivide if needed
        )
        
        if not sol.success:
            # Fallback if DOP853 fails for some reason
            sol = scipy.integrate.solve_ivp(
                f_counted,
                t_span,
                y0,
                method='RK45',
                rtol=1e-12,
                atol=1e-14
            )
            
        if not sol.success:
            # If all fails, return zero vector (invalid but safe)
            return np.zeros_like(y0, dtype=float)
            
        y_end = sol.y[:, -1]
        
        # Ensure result is finite
        if not np.all(np.isfinite(y_end)):
            return np.zeros_like(y0, dtype=float)
            
        return y_end

    except TimeoutError:
        # If we run out of time, we can't easily recover partial state from solve_ivp.
        # But this is extremely unlikely for this problem.
        return np.zeros_like(y0, dtype=float)
    except Exception:
        return np.zeros_like(y0, dtype=float)

# EVOLVE-BLOCK-END
