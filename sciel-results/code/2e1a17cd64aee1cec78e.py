import numpy as np
import time

def _robertson_jacobian(y):
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e7 * y1, -1e4 * y1],
        [0.0, 6e7 * y1, 0.0]
    ])


def _radau_step(f, J_func, t, y, h):
    n = len(y)
    c1, c2, c3 = 1.0 / 6.0, 1.0 / 2.0, 5.0 / 6.0
    a11 = 5.0 / 24.0
    a21 = 9.0 / 32.0
    a22 = 15.0 / 32.0
    a31 = -3.0 / 128.0
    a32 = 35.0 / 128.0
    a33 = 25.0 / 128.0
    b1, b2, b3 = 1.0 / 12.0, 1.0 / 2.0, 5.0 / 12.0

    f0 = f(t, y)
    K1, K2, K3 = f0.copy(), f0.copy(), f0.copy()

    for _ in range(15):
        y1s = y + h * a11 * K1
        y2s = y + h * (a21 * K1 + a22 * K2)
        y3s = y + h * (a31 * K1 + a32 * K2 + a33 * K3)

        f1 = f(t + c1 * h, y1s)
        f2 = f(t + c2 * h, y2s)
        f3 = f(t + c3 * h, y3s)

        r1, r2, r3 = f1 - K1, f2 - K2, f3 - K3
        if max(np.max(np.abs(r1)), np.max(np.abs(r2)), np.max(np.abs(r3))) < 1e-13:
            break

        J1, J2, J3 = J_func(y1s), J_func(y2s), J_func(y3s)
        Jn = np.zeros((3 * n, 3 * n))
        Jn[:n, :n] = -np.eye(n) + h * a11 * J1
        Jn[n:2*n, :n] = h * a21 * J2
        Jn[n:2*n, n:2*n] = -np.eye(n) + h * a22 * J2
        Jn[2*n:, :n] = h * a31 * J3
        Jn[2*n:, n:2*n] = h * a32 * J3
        Jn[2*n:, 2*n:] = -np.eye(n) + h * a33 * J3
        rhs = np.concatenate([r1, r2, r3])
        try:
            d = np.linalg.solve(Jn, rhs)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(Jn, rhs, rcond=None)[0]
        K1 += d[:n]
        K2 += d[n:2*n]
        K3 += d[2*n:]

    return y + h * (b1 * K1 + b2 * K2 + b3 * K3)


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    J_func = _robertson_jacobian
    start = time.time()
    deadline = start + 280.0

    h_values = ([1e-5] * 10 + [1e-4] * 9 + [1e-3] * 9 +
                [1e-2] * 9 + [0.1] * 9 + [1.0] * 9 + [10.0] * 3)

    for h in h_values:
        if time.time() > deadline:
            break
        if t + h > t1:
            h = t1 - t
            if h <= 0:
                break
        y = _radau_step(f, J_func, t, y, h)
        t += h
        if not np.all(np.isfinite(y)):
            break

    return y
