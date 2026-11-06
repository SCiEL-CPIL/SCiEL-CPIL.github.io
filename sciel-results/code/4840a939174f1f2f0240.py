import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0  # 300s limit, keep margin
    
    # Strategy 1: Assume linear autonomous system y' = J y
    # Estimate J
    # We need to be careful with call count.
    # 1 call for f(t0, y0)
    # n calls for f(t0, y0 + eps*e_i)
    # 1 call for f(t1, y0) to check autonomy? Or just assume?
    # Let's try to estimate J and check consistency.
    
    try:
        # Estimate Jacobian at t0
        y_base = f(t0, y0)
        
        # Check if y0 is zero vector? If so, J is harder to estimate from perturbations if f is linear?
        # f(t, 0) = 0 for linear homogeneous.
        # If y0 is not zero, we can use finite differences.
        
        eps = 1e-7
        J = np.zeros((n, n))
        
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps * max(1.0, abs(y0[i]))
            y_pert = y0 + e
            f_pert = f(t0, y_pert)
            J[:, i] = (f_pert - y_base) / e[i]
            
        # Check if system is autonomous: f(t1, y0) should be approx J @ y0
        # But we haven't called f(t1, y0) yet. 
        # Let's assume it is autonomous and linear for now.
        # Compute matrix exponential
        # J is 100x100. expm is O(n^3) ~ 10^6 ops. Fast.
        dt = t1 - t0
        M = expm(J * dt)
        y_sol = M @ y0
        
        # Verify solution by checking residual?
        # We can't easily check without more calls.
        # However, if the assumption is wrong, the error will be large.
        # We can do a quick check: run a few steps of RK4 and compare?
        # That costs calls.
        
        # Let's do a sanity check with 1 step of RK4 using J
        # k1 = J y0
        # k2 = J (y0 + dt/2 k1)
        # ...
        # This is cheap (no f calls, just matrix mults).
        
        # If the problem is non-linear, J is only valid locally.
        # Convection-diffusion is linear.
        
        # Let's add a fallback: if the solution seems "weird" (e.g. huge values), or if we want to be safe.
        # But 100 calls is very cheap.
        
        # To be robust, let's check if f(t1, y0) matches J@y0.
        # This costs 1 more call.
        y_check = f(t1, y0)
        if np.allclose(y_check, J @ y0, rtol=1e-5, atol=1e-8):
            # It seems autonomous and linear
            return y_sol
        else:
            # Fall back to ODE solver
            pass
    except Exception:
        pass
    
    # Strategy 2: Robust ODE Solver
    # Use LSODA or Radau
    # LSODA is good for mixed stiff/non-stiff
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=0.01 # Limit step size to ensure accuracy
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Strategy 3: High-order RK4 with many steps (fallback)
    # Use more steps than the seed program
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline - 10:
            return y
    return y
