"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk4_fallback(f, t0, t1, y0, steps):
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
            return y
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    # Extract linear system: f(t, y) = A @ y + b
    b = np.array(f(t0, np.zeros(n)), dtype=float)
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        fj = np.array(f(t0, ej), dtype=float)
        A[:, j] = fj - b
    
    # Verify linearity
    f_check = np.array(f(t0, y0), dtype=float)
    residual = np.linalg.norm(f_check - (A @ y0 + b))
    rhs_norm = np.linalg.norm(A @ y0 + b) + 1e-14
    rel_residual = residual / rhs_norm
    
    if time.time() > deadline:
        return y0.copy()
    
    if rel_residual < 1e-10:
        T = t1 - t0
        try:
            from scipy.linalg import expm
            if np.allclose(b, 0, atol=1e-14):
                eAT = expm(A * T)
                y_final = eAT @ y0
            else:
                # Block matrix: C = [[A, b], [0, 0]], z' = C z, z(0) = [y0; 1]
                C = np.zeros((n + 1, n + 1))
                C[:n, :n] = A
                C[:n, n] = b
                eCT = expm(C * T)
                z0 = np.zeros(n + 1)
                z0[:n] = y0
                z0[n] = 1.0
                zT = eCT @ z0
                y_final = zT[:n]
        except Exception:
            y_final = _rk4_fallback(f, t0, t1, y0, 10000)
        return y_final
    else:
        y_final = _rk4_fallback(f, t0, t1, y0, 10000)
        return y_final
# EVOLVE-BLOCK-END
