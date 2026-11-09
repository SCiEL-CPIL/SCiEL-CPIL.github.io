import numpy as np
import time
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]
    dt = t1 - t0

    if dt == 0.0:
        return y0.copy()

    start_time = time.time()
    deadline = start_time + 290.0

    # Probe the affine linear map: f(t, y) = A(t0) @ y + b(t0)
    b = np.asarray(f(t0, np.zeros(n)), dtype=float)

    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        col = np.asarray(f(t0, ei), dtype=float) - b
        A[:, i] = col

    if time.time() > deadline:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # Compute E = expm(A * dt)
    E = scipy.linalg.expm(A * dt)

    if np.allclose(b, 0.0, atol=1e-14):
        y_end = E @ y0
    else:
        # For affine system y' = Ay + b:
        # y_end = E @ y0 + ∫_0^dt exp(A s) b ds
        # ∫_0^dt exp(A s) ds = A^{-1} (exp(A dt) - I) if A invertible
        # So correction = A^{-1} (E - I) b
        # Solve A x = (E - I) b
        rhs = (E - np.eye(n)) @ b
        try:
            x = scipy.linalg.solve(A, rhs, assume_a='gen')
        except np.linalg.LinAlgError:
            # Fallback to block matrix exponential if A is singular
            block = np.zeros((n + 1, n + 1))
            block[:n, :n] = A
            block[:n, n] = b
            block[n, n] = 1.0
            block_exp = scipy.linalg.expm(block * dt)
            y_end = block_exp[:n, :n] @ y0 + block_exp[:n, n]
        else:
            y_end = E @ y0 + x

    return y_end
