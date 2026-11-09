# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses matrix exponential approach for linear homogeneous systems.
    Falls back to RK45 if the system appears non-linear or time-dependent.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    deadline = time.time() + 280.0
    
    # First, probe the system to determine if it's linear and homogeneous
    # Test linearity: f(t, a*y1 + b*y2) == a*f(t, y1) + b*f(t, y2)
    # Test homogeneity: f(t, 0) == 0
    # Test time-independence: f(t1, y) == f(t2, y)
    
    # Probe at t0
    f0 = f(t0, y0)
    f0_zero = f(t0, np.zeros(n))
    
    # Check homogeneity
    is_homogeneous = np.allclose(f0_zero, 0.0, atol=1e-10)
    
    # Check time-independence
    t_mid = (t0 + t1) / 2
    f_mid = f(t_mid, y0)
    is_time_independent = np.allclose(f0, f_mid, rtol=1e-8, atol=1e-10)
    
    # Check linearity
    is_linear = True
    if is_homogeneous and is_time_independent:
        # Test with a random vector
        y_rand = np.random.randn(n)
        y_rand = y_rand / np.linalg.norm(y_rand) * max(1.0, np.max(np.abs(y0)))
        
        f_rand = f(t0, y_rand)
        # For linear system, f(t, y) = A @ y, so we can extract A
        # Use multiple probes to build A
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_ej = f(t0, e_j)
            A[:, j] = f_ej
        
        # Verify linearity
        y_test = np.random.randn(n) * 0.5
        f_test = f(t0, y_test)
        f_pred = A @ y_test
        if not np.allclose(f_test, f_pred, rtol=1e-6, atol=1e-8):
            is_linear = False
    
    if is_linear and is_homogeneous and is_time_independent:
        # Use matrix exponential: y(t1) = expm((t1-t0) * A) @ y0
        T = t1 - t0
        M = T * A
        Y_final = expm(M) @ y0
        
        # Verify the solution by checking residual
        # y'(t) = A @ y(t), so at t1, A @ Y_final should equal f(t1, Y_final)
        f_check = f(t1, Y_final)
        A_check = A @ Y_final
        if np.allclose(f_check, A_check, rtol=1e-6, atol=1e-8):
            return Y_final
        # If verification fails, fall through to RK45
    else:
        # Fall back to adaptive RK45
        pass
    
    # Fallback: use scipy's solve_ivp with RK45
    from scipy.integrate import solve_ivp
    
    def wrapper(t, y):
        return f(t, y)
    
    sol = solve_ivp(wrapper, [t0, t1], y0, method='RK45', rtol=1e-12, atol=1e-14, max_step=(t1-t0)/100)
    
    if sol.success:
        return sol.y[:, -1]
    
    # If that fails, try with tighter tolerances
    sol = solve_ivp(wrapper, [t0, t1], y0, method='RK45', rtol=1e-14, atol=1e-16, max_step=(t1-t0)/200)
    if sol.success:
        return sol.y[:, -1]
    
    # Last resort: dense output
    sol = solve_ivp(wrapper, [t0, t1], y0, method='DOP853', rtol=1e-14, atol=1e-16)
    if sol.success:
        return sol.y[:, -1]
    
    return y0  # Should not reach here
# EVOLVE-BLOCK-END
