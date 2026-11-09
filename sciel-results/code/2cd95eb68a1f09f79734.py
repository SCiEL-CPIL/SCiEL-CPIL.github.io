"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Probe for linear time-invariant structure: y' = A*y + b
    # b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)

    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)

    # Verify linearity and time-invariance with a test point
    y_test = np.ones(n) * 2.0
    expected = A @ y_test + b
    actual = np.array(f(t0, y_test), dtype=float)
    if not np.allclose(expected, actual, rtol=1e-10, atol=1e-12):
        # Fallback: use scipy's solve_ivp with BDF method for stiff problems
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Also check time-invariance at a different time
    t_mid = (t0 + t1) / 2
    expected_mid = A @ y_test + b
    actual_mid = np.array(f(t_mid, y_test), dtype=float)
    if not np.allclose(expected_mid, actual_mid, rtol=1e-10, atol=1e-12):
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=dt/100)
        return sol.y[:, -1]

    # Linear time-invariant system: y' = A*y + b
    # Solution: y(t) = e^(A*(t-t0)) * y0 + integral_0^(t-t0) e^(A*s) b ds
    # For constant b: integral = (e^(A*dt) - I) * A^{-1} * b if A is invertible
    # More generally: use the augmented matrix approach
    
    # Method: augment the system with a constant 1 to handle the affine term
    # [y']   = [A  b] [y]
    # [0']   = [0  0] [1]
    # Solution: [y(t)] = expm([[A, b], [0, 0]] * dt) @ [y0; 1]
    #          [1(t)]   [1]
    
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is all zeros
    
    y_aug0 = np.zeros(n + 1)
    y_aug0[:n] = y0
    y_aug0[n] = 1.0
    
    M = expm(A_aug * dt)
    y_aug = M @ y_aug0
    
    return y_aug[:n]
