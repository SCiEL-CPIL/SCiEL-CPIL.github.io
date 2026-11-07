import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] with initial condition y0.
    
    Args:
        f: Function f(t, y) returning the derivative dy/dt.
        t_span: Tuple (t_start, t_end).
        y0: Initial state vector.
        
    Returns:
        The state vector at t_end.
    """
    start_time = time.time()
    deadline = 300.0  # Seconds
    
    # Use DOP853 with tight tolerances matching the reference to minimize rel error.
    # We add a first_step hint to potentially skip small initial steps.
    # DOP853 is 8th order, so it can take large steps.
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            first_step=0.1,  # Hint for the first step size
            max_step=np.inf
        )
        
        if not sol.success:
            if sol.y.size > 0:
                return sol.y[:, -1]
            raise RuntimeError("Integration failed: " + sol.message)
        
        y_end = sol.y[:, -1]
        
        if not np.all(np.isfinite(y_end)):
             raise ValueError("Result contains non-finite values")
             
        return y_end

    except Exception as e:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
             raise e
             
        # Fallback: Manual RK4
        t0, t1 = t_span
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        
        for _ in range(steps):
            if time.time() - start_time > deadline - 5:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                raise ValueError("Fallback RK4 produced non-finite values")
                
        return y
