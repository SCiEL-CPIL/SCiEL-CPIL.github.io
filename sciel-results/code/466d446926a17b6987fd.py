import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def validate(circles, w):
        h = 2.0 - w
        if w < 1e-6 or h < 1e-6:
            return False
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(r < -1e-9):
            return False
        if np.any(x - r < -1e-7) or np.any(x + r - w > 1e-7):
            return False
        if np.any(y - r < -1e-7) or np.any(y + r - h > 1e-7):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                if d2 < (r[i] + r[j]) ** 2 - 1e-7:
                    return False
        return True

    best_val = -1.0
    best_circles = None

    aspect_ratios = np.linspace(0.35, 0.65, 11)

    for t in aspect_ratios:
        if time.time() > deadline - 10:
            break
        w = 2.0 * t
        h = 2.0 * (1.0 - t)

        n_cols = max(1, int(np.ceil(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))

        x_grid, y_grid = np.meshgrid(
            np.linspace(0.5 / n_cols, 1.0 - 0.5 / n_cols, n_cols),
            np.linspace(0.5 / n_rows, 1.0 - 0.5 / n_rows, n_rows)
        )
        x0 = x_grid.flatten()[:N] * w
        y0 = y_grid.flatten()[:N] * h
        r0 = np.full(N, min(w / n_cols, h / n_rows) * 0.45)

        params0 = np.concatenate([x0, y0, r0])

        def penalty_obj(p, lam=50.0):
            x, y, r = p[0:N], p[N:2 * N], p[2 * N:3 * N]
            obj = -np.sum(r)
            pen = 0.0
            for i in range(N):
                v = x[i] - r[i]
                if v < 0:
                    pen += v * v
                v = x[i] + r[i] - w
                if v > 0:
                    pen += v * v
                v = y[i] - r[i]
                if v < 0:
                    pen += v * v
                v = y[i] + r[i] - h
                if v > 0:
                    pen += v * v
            for i in range(N):
                for j in range(i + 1, N):
                    d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                    md = r[i] + r[j]
                    diff = md * md - d2
                    if diff > 0:
                        pen += diff * diff
            return obj + lam * pen

        try:
            res = minimize(penalty_obj, params0, method='Powell',
                           options={'maxiter': 80, 'xtol': 1e-8, 'ftol': 1e-8})
            x, y, r = res.x[0:N], res.x[N:2 * N], res.x[2 * N:3 * N]
            circles = np.column_stack([x, y, r])
            if validate(circles, w):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_circles = circles.copy()
        except Exception:
            pass

    if best_circles is not None:
        return best_circles

    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
