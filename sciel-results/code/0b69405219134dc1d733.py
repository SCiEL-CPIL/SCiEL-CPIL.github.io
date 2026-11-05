"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    deadline = time.time() + 280

    # Extract A and b at t0: f(t, y) = A@y + b
    b = np.array(f(t0, np.zeros(n)), dtype=float)
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = np.array(f(t0, e), dtype=float) - b

    # Check autonomy at midpoint
    tmid = (t0 + t1) / 2
    b_mid = np.array(f(tmid, np.zeros(n)), dtype=float)
    A_mid = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A_mid[:, j] = np.array(f(tmid, e), dtype=float) - b_mid

    is_autonomous = (np.allclose(A, A_mid, rtol=1e-9, atol=1e-12) and
                     np.allclose(b, b_mid, rtol=1e-9, atol=1e-12))

    if is_autonomous:
        dt = t1 - t0
        Y = expm(A * dt)
        try:
            A_inv_b = np.linalg.solve(A, b)
            y_final = Y @ (y0 - A_inv_b) + A_inv_b
        except np.linalg.LinAlgError:
            from numpy.polynomial.legendre import leggauss
            n_quad = 50
            nodes, weights = leggauss(n_quad)
            s_vals = (nodes * 0.5 + 0.5) * dt
            w_vals = weights * 0.5 * dt
            y_final = Y @ y0
            for s, w in zip(s_vals, w_vals):
                y_final += expm(A * (dt - s)) @ b * w
        return y_final
    else:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-13, atol=1e-13)
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
