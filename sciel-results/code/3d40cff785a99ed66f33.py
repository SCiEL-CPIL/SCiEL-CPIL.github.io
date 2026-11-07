import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use Radau method to match the reference solver's characteristics for stiff problems
    # Tighten tolerances to reduce relative error, accepting a moderate increase in function calls
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback if Radau fails, though unlikely for this problem
        # Try BDF as another stiff solver
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    if not sol.success:
        # Last resort: explicit RK45 with tight tolerances, might be slow or inaccurate for stiff
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14
        )

    y_end = sol.y[:, -1]
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(y_end)):
        # If something went wrong, return zeros or raise? 
        # The contract says exception makes result invalid. 
        # But returning a non-finite vector also makes it invalid. 
        # Let's try to recover or just return the computed value if it's finite. 
        # If it's not finite, we can't really do much better without more context. 
        pass
        
    return y_end
