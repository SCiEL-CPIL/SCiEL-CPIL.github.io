"""ODE solver using Matrix Exponential for Linear Time-Invariant Systems."""
import numpy as np
from scipy.linalg import expm

def _is_linear_time_invariant(f, t0, t1, y0, n):
    """Check if f is linear (or affine) and time-invariant.
    Returns (b, A) if affine linear time-invariant, else None.
    """
    # Check time-invariance with a test vector
    y_test = np.random.RandomState(0).randn(n)
    y_test = y_test / np.linalg.norm(y_test)
    ft0 = f(t0, y_test)
    ft1 = f(t1, y_test)
    if not np.allclose(ft0, ft1, rtol=1e-10, atol=1e-12):
        return None  # time-dependent
    
    # Check affine: f(t, 0) should be constant (the affine part b)
    b = f(t0, np.zeros(n))
    
    # Check linearity: f(t, y1 + y2) == f(t,y1) + f(t,y2) - b
    y1 = np.random.RandomState(1).randn(n)
    y2 = np.random.RandomState(2).randn(n)
    f_y1 = f(t0, y1)
    f_y2 = f(t0, y2)
    f_y12 = f(t0, y1 + y2)
    if not np.allclose(f_y12, f_y1 + f_y2 - b, rtol=1e-8, atol=1e-10):
        return None  # not affine linear
    
    return b

def _build_matrix_A(f, t0, n, b):
    """Build the matrix A such that f(t, y) = A y + b for all y."""
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e) - b
    return A

def _verify_A(f, t0, A, b, n):
    """Verify that f(t, y) ≈ A y + b for a few random vectors."""
    for seed in [3, 4, 5]:
        y = np.random.RandomState(seed).randn(n)
        y = y / np.linalg.norm(y)
        lhs = f(t0, y)
        rhs = A @ y + b
        if not np.allclose(lhs, rhs, rtol=1e-8, atol=1e-10):
            return False
    return True

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try to detect linear time-invariant system
    try:
        b = _is_linear_time_invariant(f, t0, t1, y0, n)
        if b is not None:
            A = _build_matrix_A(f, t0, n, b)
            if _verify_A(f, t0, A, b, n):
                T = t1 - t0
                # Construct the block matrix for the affine system
                # y' = Ay + b
                # Let z = [y; 1], then z' = [A b; 0 0] z
                M = np.zeros((n + 1, n + 1))
                M[:n, :n] = A
                M[:n, n] = b
                # M[n, n] is 0, rest is 0
                
                # Compute matrix exponential
                E = expm(M * T)
                
                # Initial state in augmented space
                z0 = np.zeros(n + 1)
                z0[:n] = y0
                z0[n] = 1.0
                
                # Final state
                z_final = E @ z0
                y_final = z_final[:n]
                
                return y_final
    except Exception:
        pass

    # Fallback: If linear detection fails, use a simple high-accuracy RK45
    # This is unlikely for the specified task, but good for robustness.
    # Using solve_ivp as fallback is safer than fixed step RK4.
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, t_span, y0, method='RK45', rtol=1e-10, atol=1e-12)
    if sol.success:
        return sol.y[:, -1]
    return y0
