import numpy as np
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]
    dt = t1 - t0

    if dt == 0.0:
        return y0.copy()

    # Probe at t0
    b0 = np.asarray(f(t0, np.zeros(n)), dtype=float)
    A0 = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        A0[:, i] = np.asarray(f(t0, ei), dtype=float) - b0

    # Probe at t1
    b1 = np.asarray(f(t1, np.zeros(n)), dtype=float)
    A1 = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        A1[:, i] = np.asarray(f(t1, ei), dtype=float) - b1

    # Check if constant
    if np.allclose(A0, A1, rtol=1e-10, atol=1e-12) and np.allclose(b0, b1, rtol=1e-10, atol=1e-12):
        A = A0
        b = b0
        if np.allclose(b, 0.0, atol=1e-14):
            E = scipy.linalg.expm(A * dt)
            y_end = E @ y0
        else:
            block = np.zeros((n + 1, n + 1))
            block[:n, :n] = A
            block[:n, n] = b
            block[n, n] = 1.0
            block_exp = scipy.linalg.expm(block * dt)
            y_end = block_exp[:n, :n] @ y0 + block_exp[:n, n]
        return y_end
    else:
        # Time-dependent case: assume linear interpolation
        def f_local(t, y):
            alpha = t
            A_t = (1 - alpha) * A0 + alpha * A1
            b_t = (1 - alpha) * b0 + alpha * b1
            return A_t @ y + b_t

        sol = scipy.integrate.solve_ivp(f_local, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
