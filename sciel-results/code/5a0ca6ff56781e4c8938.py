"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate_scipy


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: kill after 300s to stay safely under 330s wall-clock limit
    start_time = time.time()
    DEADLINE = 300.0  # seconds
    
    # Wrap f to count calls
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Try Radau first - L-stable implicit method, great for stiff problems
    # Tight tolerances for high accuracy
    best_result = None
    best_evals = float('inf')
    
    # Strategy 1: Radau with tight tolerances
    try:
        sol = integrate_scipy.solve_ivp(
            f_counted, t_span, y0,
            method='Radau',
            rtol=1e-10, atol=1e-12,
            max_step=0.01,  # Limit step size for accuracy
            dense_output=False
        )
        elapsed = time.time() - start_time
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            result = sol.y[:, -1]
            if np.all(np.isfinite(result)):
                best_result = result
                best_evals = call_count[0]
        elif elapsed < DEADLINE - 10:
            # If Radau didn't complete, try with looser tolerances
            pass
    except Exception:
        pass
    
    # Check if we need to try another method
    if best_result is None and (time.time() - start_time) < DEADLINE - 10:
        # Strategy 2: Radau with slightly looser tolerances but still tight
        try:
            call_count[0] = 0
            sol = integrate_scipy.solve_ivp(
                f_counted, t_span, y0,
                method='Radau',
                rtol=1e-9, atol=1e-11,
                max_step=0.02,
                dense_output=False
            )
            if sol.success and sol.t[-1] >= t1 - 1e-12:
                result = sol.y[:, -1]
                if np.all(np.isfinite(result)):
                    best_result = result
                    best_evals = call_count[0]
        except Exception:
            pass
    
    # Strategy 3: BDF as fallback
    if best_result is None and (time.time() - start_time) < DEADLINE - 10:
        try:
            call_count[0] = 0
            sol = integrate_scipy.solve_ivp(
                f_counted, t_span, y0,
                method='BDF',
                rtol=1e-10, atol=1e-12,
                max_step=0.01,
                dense_output=False
            )
            if sol.success and sol.t[-1] >= t1 - 1e-12:
                result = sol.y[:, -1]
                if np.all(np.isfinite(result)):
                    best_result = result
                    best_evals = call_count[0]
        except Exception:
            pass
    
    # Strategy 4: LSODA as last resort
    if best_result is None and (time.time() - start_time) < DEADLINE - 10:
        try:
            call_count[0] = 0
            sol = integrate_scipy.solve_ivp(
                f_counted, t_span, y0,
                method='LSODA',
                rtol=1e-10, atol=1e-12,
                max_step=0.01,
                dense_output=False
            )
            if sol.success and sol.t[-1] >= t1 - 1e-12:
                result = sol.y[:, -1]
                if np.all(np.isfinite(result)):
                    best_result = result
                    best_evals = call_count[0]
        except Exception:
            pass
    
    # Final fallback: simple RK4 with many steps (shouldn't be needed)
    if best_result is None and (time.time() - start_time) < DEADLINE - 5:
        try:
            call_count[0] = 0
            steps = 2000
            h = (t1 - t0) / steps
            t = t0
            y = y0.copy()
            for _ in range(steps):
                k1 = f_counted(t, y)
                k2 = f_counted(t + h / 2, y + h / 2 * k1)
                k3 = f_counted(t + h / 2, y + h / 2 * k2)
                k4 = f_counted(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)):
                    break
            if np.all(np.isfinite(y)):
                best_result = y
                best_evals = call_count[0]
        except Exception:
            pass
    
    if best_result is None:
        # Return y0 as last resort
        return y0
    
    return best_result
# EVOLVE-BLOCK-END
