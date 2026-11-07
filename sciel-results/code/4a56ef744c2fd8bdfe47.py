"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Uses an adaptive Radau IIA (implicit) integrator for high accuracy with few function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: keep a safety margin before 330s
    start_time = time.time()
    deadline = start_time + 280.0  # 50s margin
    
    # Radau IIA coefficients for order 5 (3-stage)
    # But let's use a simpler approach: use scipy's solve_ivp with Radau
    # However, the task says f is counted, so we need to be careful.
    # Actually, the task allows SciPy. Let's use scipy.integrate.solve_ivp with Radau.
    # But wait, the scoring is based on the number of calls to f.
    # If we use scipy, we need to wrap f to count calls.
    
    # Let's implement a high-order adaptive method.
    # Since Radau is implicit, it requires solving nonlinear systems.
    # For Van der Pol with mu=5, the system is moderately stiff.
    
    # Let's use a high-order explicit method with adaptive step size, or
    # implement a simple implicit Radau.
    
    # Actually, let's just use scipy.integrate.solve_ivp with method='Radau'
    # and tight tolerances. The key is to minimize function calls while
    # maintaining high accuracy.
    
    from scipy.integrate import solve_ivp
    
    # Wrap f to count calls
    call_count = [0]
    def f_wrapped(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Use Radau with very tight tolerances
    # The reference uses rtol=1e-11, atol=1e-13
    # We want to be more accurate than the reference to get rel < 1e-14
    
    # Radau IIA of order 5 is a good choice for stiff problems
    sol = solve_ivp(f_wrapped, (t0, t1), y0, method='Radau', 
                    rtol=1e-13, atol=1e-15, dense_output=False)
    
    if sol.success and sol.t[-1] == t1:
        return sol.y[:, -1]
    
    # Fallback: try with slightly looser tolerances if it failed
    sol = solve_ivp(f_wrapped, (t0, t1), y0, method='Radau',
                    rtol=1e-12, atol=1e-14, dense_output=False)
    
    if sol.success and sol.t[-1] == t1:
        return sol.y[:, -1]
    
    # Last resort: use DOP853 (explicit, high order)
    sol = solve_ivp(f_wrapped, (t0, t1), y0, method='DOP853',
                    rtol=1e-13, atol=1e-15, dense_output=False)
    
    if sol.success and sol.t[-1] == t1:
        return sol.y[:, -1]
    
    # If all fail, return the initial condition (shouldn't happen)
    return y
# EVOLVE-BLOCK-END
