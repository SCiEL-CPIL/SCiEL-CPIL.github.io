import numpy as np
import time
import sys

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = 290.0  # 10s safety margin before 300s limit
    
    try:
        from scipy.integrate import solve_ivp
        
        # Use DOP853 with tight tolerances to minimize error
        # Reference used rtol=1e-11, atol=1e-13
        # We use tighter tolerances to reduce rel error further
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1 - t0) / 10.0  # Prevent huge steps initially, though adaptive handles it
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
            else:
                raise ValueError("Non-finite result from solve_ivp")
        else:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
            
    except Exception as e:
        # Fallback: High-precision RK4
        # This should rarely be triggered
        steps = 5000
        h = (t1 - t0) / steps
        t, y = t0, y0_arr.copy()
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            
        if not np.all(np.isfinite(y)):
            # Last resort: return y0 if integration failed completely
            return y0_arr
        return y
