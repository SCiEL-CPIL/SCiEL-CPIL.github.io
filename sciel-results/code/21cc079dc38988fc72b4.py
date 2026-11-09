"""ODE solver: Heat2D 10x10 – integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import time


def _rk4(f, t0, t1, y0, steps, deadline):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def _backward_euler(f, t0, t1, y0, steps, deadline):
    # Implicit Euler with a finite-difference Jacobian and a few Newton iterations (handles stiff problems).
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        if time.time() > deadline:
            break
        z = y.copy()
        for _newton in range(4):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt_total = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    # Strategy: Assume linear autonomous ODE y' = L y
    # Reconstruct L using N calls
    L = np.zeros((N, N), dtype=float)
    for i in range(N):
        if time.time() > deadline:
            break
        e_i = np.zeros(N)
        e_i[i] = 1.0
        # Call f at t0 with e_i
        L[:, i] = f(t0, e_i)
        
    # Check linearity and time-invariance
    # Test with a deterministic vector and a different time
    test_vec = np.ones(N)
    test_vec[0] = 2.0
    test_vec[1] = -1.0 if N > 1 else 0.0
    test_time = t0 + 0.5 * dt_total if dt_total > 0 else t0
    f_test = f(test_time, test_vec)
    L_test = L @ test_vec
    
    # Check time invariance by comparing f(t0, v) and f(t1, v) roughly?
    # The eigenvalues of L determine stability.
    # If max norm difference is small, assume linear autonomous.
    norm_L_test = np.linalg.norm(L_test)
    if norm_L_test < 1e-10:
        # If L_test is near zero, check if f_test is also near zero
        if np.linalg.norm(f_test) < 1e-10:
            is_linear_autonomous = True
        else:
            is_linear_autonomous = False
    else:
        is_linear_autonomous = np.linalg.norm(f_test - L_test) < 1e-8 * (1 + norm_L_test)

    if is_linear_autonomous:
        # Use exact exponential solution
        try:
            M = scipy.linalg.expm(L * dt_total)
            y_final = M @ y0
            if np.all(np.isfinite(y_final)):
                return y_final
        except Exception:
            pass # Fall through to generic solver

    # Fallback: Use a robust implicit solver if linear assumption fails
    # Given the task is "Heat Equation", the linear case is 99.9% probability.
    # If the check failed, something is weird. 
    # Let's try a standard RK4 with adaptive steps or just a fixed step BDF.
    
    # Simple fallback: RK4 with 1000 steps (4000 calls)
    # Total calls ~ 100 + 4000 = 4100.
    steps = 1000
    y = _rk4(f, t0, t1, y0, steps, deadline)
    if y is None:
        y = _backward_euler(f, t0, t1, y0, steps, deadline)
    return y
# EVOLVE-BLOCK-END
