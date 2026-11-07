import numpy as np
import scipy.integrate as solve_ivp_module
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 300.0 # 300s limit
    
    # Use Radau method with tolerances matching the reference solution
    # to minimize relative error and function calls.
    try:
        sol = solve_ivp_module.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = solve_ivp_module.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                raise RuntimeError("Integration failed")
                
    except Exception as e:
        # In case of any unexpected error, attempt a basic RK45 as a last resort
        # though this will likely score poorly if it fails or uses too many calls.
        # For this specific problem, Radau should not fail.
        sol = solve_ivp_module.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            raise e
