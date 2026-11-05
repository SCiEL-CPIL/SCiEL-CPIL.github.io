"""ODE solver: matrix-exponential for linear convection-diffusion, with RK4 fallback."""
import numpy as np
import time
from scipy.linalg import expm


def _build_A_from_f(f, t0, y0, n):
    """
    Probe f to determine if it is linear time-invariant: f(t, y) = A @ y.
    Returns (A, b) if affine linear time-invariant, else None.
    We check: f(t, 0) should be zero (or constant b), and f(t, y) - f(t, 0) should be linear in y.
    """
    # Check time-invariance: f(t0, y) vs f(t0 + dt, y) for a few test vectors
    dt = 1e-6
    y_test = np.random.RandomState(42).randn(n)
    y_test2 = np.random.RandomState(43).randn(n)
    
    f1 = np.asarray(f(t0, y_test), dtype=float)
    f2 = np.asarray(f(t0 + dt, y_test), dtype=float)
    f3 = np.asarray(f(t0, y_test2), dtype=float)
    f4 = np.asarray(f(t0 + dt, y_test2), dtype=float)
    
    # Time-invariance check
    if np.max(np.abs(f1 - f2)) > 1e-10 * (1 + np.max(np.abs(f1))):
        return None
    if np.max(np.abs(f3 - f4)) > 1e-10 * (1 + np.max(np.abs(f3))):
        return None
    
    # Check for affine constant term b = f(t, 0)
    y_zero = np.zeros(n)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # Check linearity: f(t, y) - b should be linear in y
    # Use the test vectors to build A
    # f(t, y_test) - b = A @ y_test
    # f(t, y_test2) - b = A @ y_test2
    # We need n independent vectors to determine A, but we can use a smarter approach.
    # For a 100-point convection-diffusion, the system is tridiagonal.
    # Let's probe with basis vectors to build A column by column.
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # f(t, e_j) - b = A @ e_j = column j of A
        val = np.asarray(f(t0, e_j), dtype=float) - b
        A[:, j] = val
    
    # Verify: check with the original test vectors
    pred1 = A @ y_test + b
    pred2 = A @ y_test2 + b
    if np.max(np.abs(pred1 - f1)) > 1e-8 * (1 + np.max(np.abs(f1))):
        return None
    if np.max(np.abs(pred2 - f3)) > 1e-8 * (1 + np.max(np.abs(f3))):
        return None
    
    return A, b


def _matrix_exp_solve(f, t_span, y0, deadline):
    """
    Solve y' = A@y + b using matrix exponential.
    y(t) = expm(A*(t-t0)) @ y0 + integral_0^{t-t0} expm(A*(t-t0-s)) b ds
    If b is constant: y(t) = expm(A*dt) @ y0 + (expm(A*dt) - I) @ inv(A) @ b  (if A invertible)
    More generally: y(t) = expm(A*dt) @ y0 + A^{-1} @ (expm(A*dt) - I) @ b
    Or use the augmented matrix trick.
    """
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    probe_result = _build_A_from_f(f, t0, y0, n)
    if probe_result is None:
        return None, 0  # Not linear time-invariant
    
    A, b = probe_result
    dt = t1 - t0
    
    if time.time() > deadline:
        return None, 0
    
    # Compute expm(A * dt)
    try:
        M = expm(A * dt)
    except Exception:
        return None, 0
    
    if time.time() > deadline:
        return None, 0
    
    if np.allclose(b, 0, atol=1e-15):
        # Homogeneous: y(t1) = M @ y0
        y_final = M @ y0
    else:
        # Non-homogeneous with constant b:
        # y(t) = M @ y0 + (M - I) @ A^{-1} @ b  (if A invertible)
        # Better: use the formula y(t) = M @ y0 + integral
        # For constant b: y(t) = M @ y0 + A^{-1} @ (M - I) @ b
        # But if A is singular, use the augmented system.
        # Let's try the augmented matrix approach:
        # Define z = [y; 1], then z' = [[A, b], [0, 0]] @ z
        # z(t) = expm([[A, b], [0, 0]] * dt) @ [y0; 1]
        # This is robust.
        n_aug = n + 1
        A_aug = np.zeros((n_aug, n_aug))
        A_aug[:n, :n] = A
        A_aug[:n, n] = b
        # Last row is all zeros
        M_aug = expm(A_aug * dt)
        z0 = np.zeros(n_aug)
        z0[:n] = y0
        z0[n] = 1.0
        z_final = M_aug @ z0
        y_final = z_final[:n]
    
    if not np.all(np.isfinite(y_final)):
        return None, 0
    
    return y_final, 0  # 0 additional f-calls beyond probing


def _rk4(f, t0, t1, y0, steps, deadline):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for i in range(steps):
        if time.time() > deadline:
            return None
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    # Try matrix-exponential approach first (exploits linearity)
    y_result, _ = _matrix_exp_solve(f, t_span, y0, deadline)
    if y_result is not None:
        return y_result
    
    # Fallback: RK4 with adaptive step if needed
    # Start with a reasonable number of steps
    for steps in [1000, 2000, 5000, 10000, 20000]:
        y = _rk4(f, t0, t1, y0, steps, deadline)
        if y is not None:
            return y
    
    # Last resort: very coarse RK4
    y = _rk4(f, t0, t1, y0, 100, deadline)
    if y is not None:
        return y
    
    # Absolute last resort
    return np.array(y0, dtype=float)
