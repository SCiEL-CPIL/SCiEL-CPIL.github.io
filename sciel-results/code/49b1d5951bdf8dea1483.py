import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    # Use scipy.integrate.ode with dopri853
    # This uses a different internal implementation/interface than solve_ivp
    ode_solver = scipy.integrate.ode(f)
    ode_solver.set_integrator('dopri853', rtol=1e-11, atol=1e-13)
    ode_solver.set_initial_value(y0, t_span[0])
    
    # Integrate
    # The 'integrate' method advances to the target time
    # It returns the state at the target time
    y_end = ode_solver.integrate(t_span[1])
    
    if y_end is None:
        raise RuntimeError("Integration failed")
        
    return y_end
