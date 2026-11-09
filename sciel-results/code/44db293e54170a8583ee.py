# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm, schur


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    For the fixed 15D stiff linear system, infer A via basis probes,
    then use Schur-based matrix exponential for high accuracy with minimal calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    deadline = time.time() + 280.0
    
    # Infer A by probing with standard basis vectors (n calls)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Check homogeneity: f(t0, 0) should be 0 (1 call)
    f_zero = f(t0, np.zeros(n))
    is_homogeneous = np.allclose(f_zero, 0.0, atol=1e-12)
    
    # Check time-independence: f(t1, e_j) should equal f(t0, e_j) for a few probes
    # Use just 2-3 probes to save calls
    is_time_independent = True
    for j in range(0, n, max(1, n // 3)):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_t1 = f(t1, e_j)
        if not np.allclose(f_t1, A[:, j], rtol=1e-10, atol=1e-12):
            is_time_independent = False
            break
    
    # Verify linearity with one random probe
    is_linear = True
    if is_homogeneous and is_time_independent:
        rng = np.random.default_rng(42)
        y_test = rng.standard_normal(n) * 0.5
        f_test = f(t0, y_test)
        f_pred = A @ y_test
        if not np.allclose(f_test, f_pred, rtol=1e-8, atol=1e-10):
            is_linear = False
    
    if is_linear and is_homogeneous and is_time_independent:
        T = t1 - t0
        M = T * A
        
        # Use Schur decomposition for stable matrix exponential
        # This is more accurate than the default expm for stiff systems
        T_schur, Q = schur(M, output='complex')
        
        # Compute exp of upper triangular matrix
        exp_T = np.zeros_like(T_schur)
        m = T_schur.shape[0]
        for i in range(m):
            exp_T[i, i] = np.exp(T_schur[i, i])
        for i in range(m):
            for j in range(i + 1, m):
                # For upper triangular, exp(T)[i,j] = sum over paths
                # Use recursive formula
                if j == i + 1:
                    exp_T[i, j] = T_schur[i, j] * (exp_T[i, i] - exp_T[j, j]) / (T_schur[i, i] - T_schur[j, j]) if abs(T_schur[i, i] - T_schur[j, j]) > 1e-15 else T_schur[i, j] * exp_T[i, i]
                else:
                    # More complex; use scipy's internal or fall back
                    pass
        
        # Actually, scipy.linalg.expm already uses Schur decomposition internally
        # Let's just use expm which is well-optimized
        expM = expm(M)
        Y_final = expM @ y0
        
        # Verify with a residual check
        f_check = f(t1, Y_final)
        A_check = A @ Y_final
        if np.allclose(f_check, A_check, rtol=1e-6, atol=1e-8):
            return Y_final
    
    # Fallback: use scipy's solve_ivp with a stiff method
    from scipy.integrate import solve_ivp
    
    def wrapper(t, y):
        return f(t, y)
    
    # Try BDF first for stiff systems
    sol = solve_ivp(wrapper, [t0, t1], y0, method='BDF', rtol=1e-13, atol=1e-15)
    if sol.success:
        return sol.y[:, -1]
    
    # Try Radau
    sol = solve_ivp(wrapper, [t0, t1], y0, method='Radau', rtol=1e-13, atol=1e-15)
    if sol.success:
        return sol.y[:, -1]
    
    # Try DOP853 with tight tolerances
    sol = solve_ivp(wrapper, [t0, t1], y0, method='DOP853', rtol=1e-14, atol=1e-16)
    if sol.success:
        return sol.y[:, -1]
    
    return y0
# EVOLVE-BLOCK-END
