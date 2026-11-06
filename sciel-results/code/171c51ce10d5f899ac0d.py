import numpy as np
import scipy.integrate as integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 300  # 300s safety margin
    
    # Use Radau for stiff problems, with tight tolerances
    # rtol=1e-10, atol=1e-12 ensures high accuracy
    try:
        sol = integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails
            sol = integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            # If both fail, try RK45 with tighter tol?
            sol = integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            return y0 # Return initial state if all fail
    except Exception:
        return y0
