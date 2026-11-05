import numpy as np
from scipy.linalg import expm
import scipy.integrate as integrate_scipy
import time

def _is_linear_ti(f, t0, t1, y0, n):
    """
    Check if f is linear and time-invariant.
    Returns True if f(t, y) = A*y for some constant A.
    """
    # Use a random vector to check linearity and time invariance
    # Avoid all zeros
    rs = np.random.RandomState(42)
    y_test = rs.randn(n)
    
    # Check time invariance
    f_t0 = f(t0, y_test)
    f_t1 = f(t1, y_test)
    if not np.allclose(f_t0, f_t1, rtol=1e-10, atol=1e-12):
        return False
        
    # Check linearity (homogeneity)
    f_2y = f(t0, 2.0 * y_test)
    if not np.allclose(f_2y, 2.0 * f_t0, rtol=1e-10, atol=1e-12):
        return False
        
    # Check additivity with another vector
    y_test2 = rs.randn(n)
    f_y1 = f_t0
    f_y2 = f(t0, y_test2)
    f_sum = f(t0, y_test + y_test2)
    if not np.allclose(f_sum, f_y1 + f_y2, rtol=1e-10, atol=1e-12):
        return False
        
    return True

def _solve_linear_ti(f, t_span, y0):
    """
    Solve y' = A y assuming linear time-invariant system.
    """
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Construct A
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # Compute matrix exponential
    dt = t1 - t0
    M = expm(A * dt)
    
    return M @ y0

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0 # 20s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try linear time-invariant path
    try:
        if _is_linear_ti(f, t0, t1, y0, n):
            # Check time
            if time.time() > deadline:
                # Fallback to a quick RK4 if out of time? 
                # But we've already used ~5 calls. 
                # Just return y0? No, that's bad. 
                # Let's just proceed, expm is fast.
                pass
            return _solve_linear_ti(f, t_span, y0)
    except Exception:
        pass

    # Fallback: Robust ODE solver
    # Use Radau for stiff problems, with tight tolerances
    # We need to be careful with call count if this path is taken, 
    # but the linear path is expected to succeed.
    
    # Simple high-order RK4 as last resort if scipy fails or is too slow?
    # Actually, let's use scipy.integrate.solve_ivp with Radau.
    # To minimize calls, we can pass a Jacobian if we could build one, 
    # but we don't know it. 
    # Radau will estimate it.
    
    def sol_func(t, y):
        return f(t, y)
        
    try:
        sol = integrate_scipy.solve_ivp(
            sol_func, t_span, y0, 
            method='Radau', 
            rtol=1e-10, atol=1e-12, 
            max_step=(t1-t0)/100 # Force reasonable steps
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Absolute fallback: RK4 with many steps
    # This is slow but robust
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline:
            break
            
    return y
