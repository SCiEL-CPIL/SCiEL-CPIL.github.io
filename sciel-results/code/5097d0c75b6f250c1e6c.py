"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use Radau method for stiff problems
    # High precision tolerances to maximize accuracy score
    rtol = 1e-12
    atol = 1e-14
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=rtol, 
            atol=atol,
            t_eval=[t1]
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='BDF', 
                rtol=rtol, 
                atol=atol,
                t_eval=[t1]
            )
            
        return sol.y[:, -1]
    except Exception:
        # Last resort fallback to LSODA
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='LSODA', 
            rtol=1e-10, 
            atol=1e-12,
            t_eval=[t1]
        )
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
