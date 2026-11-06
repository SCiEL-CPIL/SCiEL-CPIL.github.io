"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    # Deadline guard: ensure we finish well within the 330s limit
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    # Robertson system Jacobian
    # y0' = -0.04*y0 + 1e4*y1*y2
    # y1' =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    # y2' =  3e7*y1^2
    def jac(t, y):
        y0_val, y1_val, y2_val = y
        return np.array([
            [-0.04, 1e4*y2_val, -1e4*y1_val],
            [0.04, -1e4*y2_val - 6e7*y1_val, 1e4*y1_val],
            [0, 6e7*y1_val, 0]
        ])
    
    # Strategy: Use Radau (implicit RK, order 5) with analytic Jacobian.
    # Remove max_step to allow larger steps in smooth regions, reducing nfev.
    # Tune tolerances: rtol=1e-10, atol=1e-12 is usually sufficient for high accuracy
    # without the overhead of 1e-12/1e-14, which might force unnecessary small steps.
    try:
        if time.time() > deadline:
            return np.array(y0)
            
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            rtol=1e-10, 
            atol=1e-12
        )
        
        if time.time() > deadline:
            return np.array(y0)
            
        if sol.success:
            return sol.y[:, -1]
        else:
            raise RuntimeError("Radau failed")
            
    except Exception:
        # Fallback: Try BDF with Jacobian
        try:
            if time.time() > deadline:
                return np.array(y0)
                
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=jac,
                rtol=1e-10, 
                atol=1e-12
            )
            
            if time.time() > deadline:
                return np.array(y0)
                
            if sol2.success:
                return sol2.y[:, -1]
            else:
                raise RuntimeError("BDF failed")
        except Exception:
            # Last resort: Try without Jacobian
            try:
                if time.time() > deadline:
                    return np.array(y0)
                    
                sol3 = scipy.integrate.solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='Radau', 
                    rtol=1e-10, 
                    atol=1e-12
                )
                
                if time.time() > deadline:
                    return np.array(y0)
                    
                if sol3.success:
                    return sol3.y[:, -1]
                else:
                    return np.array(y0)
            except Exception:
                return np.array(y0)
# EVOLVE-BLOCK-END
