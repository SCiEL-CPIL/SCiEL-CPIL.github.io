# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm, expm_multiply


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses matrix exponential approach for linear homogeneous systems.
    Falls back to high-precision ODE solver if the system appears non-linear or time-dependent.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0
    
    # First, probe the system to determine if it's linear and homogeneous
    # Test homogeneity: f(t, 0) == 0
    # Test time-independence: f(t1, y) == f(t2, y)
    
    # Probe at t0
    f0 = f(t0, y0)
    f0_zero = f(t0, np.zeros(n))
    
    # Check homogeneity
    is_homogeneous = np.allclose(f0_zero, 0.0, atol=1e-12)
    
    # Check time-independence
    t_mid = (t0 + t1) / 2
    f_mid = f(t_mid, y0)
    is_time_independent = np.allclose(f0, f_mid, rtol=1e-10, atol=1e-12)
    
    # Check linearity by extracting A
    is_linear = True
    if is_homogeneous and is_time_independent:
        # Extract A by probing with basis vectors
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_ej = f(t0, e_j)
            A[:, j] = f_ej
        
        # Verify linearity with a random test vector
        y_test = np.random.randn(n) * 0.5
        f_test = f(t0, y_test)
        f_pred = A @ y_test
        if not np.allclose(f_test, f_pred, rtol=1e-8, atol=1e-10):
            is_linear = False
    
    if is_linear and is_homogeneous and is_time_independent:
        # Use matrix exponential: y(t1) = expm((t1-t0) * A) @ y0
        T = t1 - t0
        M = T * A
        
        # Use expm_multiply for better numerical stability on stiff systems
        # This computes expm(M) @ y0 without explicitly forming the full matrix exponential
        Y_final = expm_multiply(M, y0)
        
        # Verify the solution by checking residual
        # y'(t) = A @ y(t), so at t1, A @ Y_final should equal f(t1, Y_final)
        f_check = f(t1, Y_final)
        A_check = A @ Y_final
        if np.allclose(f_check, A_check, rtol=1e-8, atol=1e-10):
            return Y_final
        # If verification fails, fall through to fallback
    
    # Fallback: use scipy's solve_ivp with high-precision methods
    from scipy.integrate import solve_ivp
    
    def wrapper(t, y):
        return f(t, y)
    
    # Try Radau (implicit, good for stiff problems) with tight tolerances
    sol = solve_ivp(wrapper, [t0, t1], y0, method='Radau', rtol=1e-14, atol=1e-16)
    
    if sol.success:
        return sol.y[:, -1]
    
    # Try BDF (another implicit method for stiff problems)
    sol = solve_ivp(wrapper, [t0, t1], y0, method='BDF', rtol=1e-14, atol=1e-16)
    
    if sol.success:
        return sol.y[:, -1]
    
    # Try RK45 as last resort
    sol = solve_ivp(wrapper, [t0, t1], y0, method='RK45', rtol=1e-12, atol=1e-14)
    
    if sol.success:
        return sol.y[:, -1]
    
    return y0  # Should not reach here
# EVOLVE-BLOCK-END
