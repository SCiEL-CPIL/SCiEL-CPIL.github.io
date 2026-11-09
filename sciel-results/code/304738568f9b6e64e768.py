import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    ODE integrator for the harmonic oscillator.
    
    The problem is: y0' = y1, y1' = -y0, with y(0) = [1, 0], t in [0, 10].
    The exact solution is y(t) = [cos(t), -sin(t)].
    
    We use scipy's DOP853 solver to minimize relative error while keeping
    the function call count low.
    """
    t_start, t_end = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances to get near-machine-precision results
    # This minimizes relative error while keeping call count reasonable
    sol = solve_ivp(
        f, 
        (t_start, t_end), 
        y0_arr,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        dense_output=False
    )
    
    # Return the final state
    return sol.y[:, -1]
