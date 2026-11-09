import numpy as np

# Radau IIA 3-stage coefficients (order 3, L-stable, stiffly accurate)
_C = np.array([0.1478446090758688, 0.4496230468654178, 0.9025323440587134])
_A = np.array([
    [0.0781677555976724, 0.0, 0.0],
    [0.0504692267323607, 0.1492318735167891, 0.0],
    [0.0549372709204120, 0.1634856567295209, 0.2579094164087805],
])
_B = np.array([0.0549372709204120, 0.1634856567295209, 0.2579094164087805])


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    steps = 200
    h = (t1 - t0) / steps
    t = t0
    I = np.eye(n)
    for _ in range(steps):
        hs = min(h, t1 - t)
        if hs <= 0.0:
            break
        # Stage values (collocation nodes)
        Y = np.tile(y, (3, 1)).copy()
        F = np.empty((3, n))
        J = np.empty((3, n, n))
        # Initial stage derivatives and finite-difference Jacobians (reused across Newton iters; exact for linear f)
        for k in range(3):
            tk = t + _C[k] * hs
            F[k] = f(tk, Y[k])
            Jk = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(Y[k, j]))
                Jk[:, j] = (f(tk, Y[k] + e) - F[k]) / e[j]
            J[k] = Jk
        # Newton iteration for the stage equations
        for it in range(60):
            G = np.empty(3 * n)
            for i in range(3):
                rhs = y.copy()
                for j in range(3):
                    rhs += hs * _A[i, j] * F[j]
                G[i * n:(i + 1) * n] = Y[i] - rhs
            JB = np.empty((3 * n, 3 * n))
            for i in range(3):
                for k in range(3):
                    block = -hs * _A[i, k] * J[k]
                    if i == k:
                        block = block + I
                    JB[i * n:(i + 1) * n, k * n:(k + 1) * n] = block
            try:
                stepv = np.linalg.solve(JB, -G)
            except np.linalg.LinAlgError:
                stepv = -0.01 * G
            Y = Y + stepv.reshape(3, n)
            for k in range(3):
                F[k] = f(t + _C[k] * hs, Y[k])
            if np.linalg.norm(stepv) <= 1e-12 * (1.0 + np.linalg.norm(Y)):
                break
        # Stiffly-accurate update
        y = y + hs * np.sum(_B[:, None] * F, axis=0)
        t = t + hs
    return y
