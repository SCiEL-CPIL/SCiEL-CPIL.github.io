# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: kill after ~300s to stay within 330s wall-clock
    deadline = time.time() + 290.0
    
    # Strategy:
    # 1. Try to identify if the system is linear with constant coefficients: y' = A*y + b
    #    If so, use the exact matrix exponential solution.
    # 2. If not, use a robust implicit solver (backward Euler or Radau via solve_ivp)
    
    # Probe for linearity: y' = A*y + b
    # Evaluate b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # Evaluate A columns: A[:, j] = (f(t0, e_j) - b) / 1.0  (since e_j has 1 at j)
    # Actually, f(t, y) = A*y + b => f(t, e_j) = A*e_j + b = A[:, j] + b
    # So A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.asarray(f(t0, e_j), dtype=float) - b
    
    # Verify linearity and constant coefficients with a test point
    y_test = np.ones(n) * 0.5
    f_test = np.asarray(f(t0, y_test), dtype=float)
    f_pred = A @ y_test + b
    if not np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-12):
        # Not linear with constant coefficients, or our probe was off
        # Try a different t to check time-invariance
        t_mid = 0.5 * (t0 + t1)
        y_test2 = np.ones(n) * 0.3
        f_test2 = np.asarray(f(t_mid, y_test2), dtype=float)
        f_pred2 = A @ y_test2 + b
        if not np.allclose(f_test2, f_pred2, rtol=1e-10, atol=1e-12):
            # Not a constant-coefficient linear system. Use implicit ODE solver.
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
            else:
                # Fallback to backward Euler
                return _backward_euler(f, t0, t1, y0, 1000)
    
    # It's a linear constant-coefficient system: y' = A*y + b
    # Solution: y(t) = expm(A*(t-t0)) * y0 + integral_0^{t-t0} expm(A*s) * b ds
    # For constant b: y(t) = expm(A*T) * y0 + (expm(A*T) - I) * A^{-1} * b  (if A invertible)
    # More generally: y(t) = expm(A*T) * y0 + ∫_0^T expm(A*s) ds * b
    # We can compute the integral using: ∫_0^T expm(A*s) ds = A^{-1} (expm(A*T) - I) if A is invertible
    # Or use block matrix trick: expm([[A, b],[0, 0]] * T) gives the combined solution
    
    T = t1 - t0
    
    # Use the block matrix approach for robustness
    # M = [[A, b], [0, 0]]  (size (n+1) x (n+1))
    # Then expm(M * T) * [y0; 1] = [y(T); 1]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    # M[n, :] remains zeros
    
    vec = np.zeros(n + 1)
    vec[:n] = y0
    vec[n] = 1.0
    
    MT = M * T
    exp_MT = expm(MT)
    result_vec = exp_MT @ vec
    y_end = result_vec[:n]
    
    return y_end


def _backward_euler(f, t0, t1, y0, steps):
    """Fallback implicit Euler solver."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        z = y.copy()
        for _newton in range(5):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            try:
                step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                step = 0
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y
# EVOLVE-BLOCK-END
