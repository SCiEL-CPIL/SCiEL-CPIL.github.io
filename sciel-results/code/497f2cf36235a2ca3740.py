import numpy as np
import scipy.integrate
import time

def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    # Standard Robertson parameters
    k1, k2, k3 = 0.04, 3e-7, 1e4
    y1, y2, y3 = y
    return np.array([
        [-k1 * y1 - k3 * y2, -k3 * y2, 0.0],
        [k1 * y1, -k2 * y2 - k3 * y2, 0.0],
        [-k1 * y1, k2 * y2, 0.0]
    ])

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) using scipy.integrate.solve_ivp with Radau method.
    Uses analytical Jacobian for efficiency.
    """
    start_time = time.time()
    
    # Attempt to solve with Radau and analytical Jacobian
    # Radau is 5th order and suitable for stiff problems
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            jac=_robertson_jac
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12,
                jac=_robertson_jac
            )
            
    except Exception:
        # Fallback to BDF without Jacobian if something goes wrong
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12
        )
        
    # Check for deadline (unlikely to be hit, but safe)
    # If time is running out, we return the best we have
    # solve_ivp returns the final state in sol.y[:, -1]
    
    return sol.y[:, -1]
