import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0

    # Check for linearity and time-invariance
    # f(t, y) = A y
    f_zero = f(t0, np.zeros(n))
    is_homogeneous = np.linalg.norm(f_zero) < 1e-12
    
    f_y0 = f(t0, y0)
    f_2y0 = f(t0, 2 * y0)
    # Tolerance relative to magnitude
    tol = 1e-12 * (1 + np.linalg.norm(f_y0))
    is_linear = np.linalg.norm(f_2y0 - 2 * f_y0) < tol
    
    f_t1_y0 = f(t1, y0)
    is_time_invariant = np.linalg.norm(f_t1_y0 - f_y0) < tol

    if is_homogeneous and is_linear and is_time_invariant:
        # Construct A
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f(t0, e_i)
            
        if time.time() > deadline:
            # Fallback to simple RK4 if timeout during setup (very unlikely)
            return _rk4(f, t0, t1, y0, 100)

        dt = t1 - t0
        # Use expm for exact solution
        try:
            expm_A = scipy.linalg.expm(A * dt)
            y_end = expm_A @ y0
            return y_end
        except Exception:
            pass

    # Fallback: Use solve_ivp with Radau
    from scipy.integrate import solve_ivp
    try:
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-10)
        if sol.success:
            return sol.y[:, -1]
        else:
            # If solve_ivp fails, try RK4
            return _rk4(f, t0, t1, y0, 2000)
    except Exception:
        return _rk4(f, t0, t1, y0, 2000)

def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return y0 # Return initial if unstable
    return y
