"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 120.0


def _aspect_to_wh(r):
    h = 2.0 / (1.0 + r)
    w = 2.0 - h
    return w, h


def _make_initial(n, w, h):
    best = None
    for rows in range(1, n + 1):
        cols = int(np.ceil(n / rows))
        r_x = w / (cols * 2.0)
        r_y = h / (rows * np.sqrt(3.0))
        r = min(r_x, r_y)
        if r <= 0:
            continue
        pts = []
        for j in range(rows):
            if (j + 1) * cols <= n:
                c_in = cols
            else:
                c_in = n - j * cols
            if c_in <= 0:
                continue
            off = 0.0
            if j % 2 == 1:
                off = r
            total_width = (c_in - 1) * 2 * r
            x0 = (w - total_width) / 2.0 + off
            y = (j + 0.5) * (h / rows)
            for i in range(c_in):
                x = x0 + i * 2 * r
                if x < r:
                    x = r
                if x > w - r:
                    x = w - r
                if y < r:
                    y = r
                if y > h - r:
                    y = h - r
                pts.append([x, y, r])
        if len(pts) == n:
            score = sum(p[2] for p in pts)
            if best is None or score > best[0]:
                best = (score, pts)
    if best is None:
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(n):
            j = i // side
            ii = i % side
            x = (ii + 0.5) * w / side
            y = (j + 0.5) * h / side
            r = min(w / side, h / side) * 0.5 - 1e-3
            pts.append([x, y, r])
        return np.array(pts)
    return np.array(best[1])


def _objective_and_grad(vars_, n, w, h):
    x = vars_[:n]
    y = vars_[n:2*n]
    r = vars_[2*n:3*n]
    obj = -np.sum(r)
    grad = np.zeros(3 * n)
    grad[2*n:3*n] = -1.0
    pen = 0.0
    pen_grad = np.zeros(3 * n)
    M = 1e4
    for i in range(n):
        v = r[i] - x[i]
        if v > 0:
            pen += M * v * v
            pen_grad[i] += 2 * M * v
            pen_grad[2*n + i] += 2 * M * v
        v = x[i] - (w - r[i])
        if v > 0:
            pen += M * v * v
            pen_grad[i] += 2 * M * v
            pen_grad[2*n + i] += -2 * M * v
        v = r[i] - y[i]
        if v > 0:
            pen += M * v * v
            pen_grad[n + i] += 2 * M * v
            pen_grad[2*n + i] += 2 * M * v
        v = y[i] - (h - r[i])
        if v > 0:
            pen += M * v * v
            pen_grad[n + i] += 2 * M * v
            pen_grad[2*n + i] += -2 * M * v
        if r[i] < 0:
            pen += M * r[i] * r[i]
            pen_grad[2*n + i] += 2 * M * r[i]
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            s = r[i] + r[j]
            v = s - d
            if v > 0:
                pen += M * v * v
                f = 2 * M * v
                gx_i = -dx / d
                gy_i = -dy / d
                gx_j = dx / d
                gy_j = dy / d
                pen_grad[i] += f * gx_i
                pen_grad[n + i] += f * gy_i
                pen_grad[2*n + i] += f
                pen_grad[j] += f * gx_j
                pen_grad[n + j] += f * gy_j
                pen_grad[2*n + j] += f
    return obj + pen, grad + pen_grad


def _validate(arr, w, h, tol=1e-6):
    n = len(arr)
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r < -tol):
        return False
    r = np.maximum(r, 0.0)
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _separate(arr, w, h, iters=200):
    n = len(arr)
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = np.maximum(arr[:, 2], 1e-9).copy()
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                s = r[i] + r[j]
                if d < s:
                    push = (s - d) / 2.0
                    ux = dx / d
                    uy = dy / d
                    x[i] += push * ux
                    y[i] += push * uy
                    x[j] -= push * ux
                    y[j] -= push * uy
                    moved = True
        for i in range(n):
            if x[i] < r[i]:
                x[i] = r[i]
            if x[i] > w - r[i]:
                x[i] = w - r[i]
            if y[i] < r[i]:
                y[i] = r[i]
            if y[i] > h - r[i]:
                y[i] = h - r[i]
        if not moved:
            break
    return np.column_stack([x, y, r])


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - 5.0
    best_arr = None
    best_val = -np.inf
    best_wh = (1.0, 1.0)

    aspects = [1.0, 1.1, 0.9, 1.25, 0.75, 1.5, 1.8, 2.2]

    for r_ratio in aspects:
        if time.time() > deadline:
            break
        w, h = _aspect_to_wh(r_ratio)
        init = _make_initial(N, w, h)
        init = _separate(init, w, h, iters=100)

        if _validate(init, w, h):
            val = np.sum(init[:, 2])
            if val > best_val:
                best_val = val
                best_arr = init.copy()
                best_wh = (w, h)

        vars0 = init.flatten()
        try:
            res = minimize(
                _objective_and_grad,
                vars0,
                args=(N, w, h),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 4000}
            )
        except Exception:
            continue

        if time.time() > deadline:
            break

        arr = res.x.reshape(N, 3)
        arr[:, 2] = np.maximum(arr[:, 2], 0.0)
        arr = _separate(arr, w, h, iters=300)

        if _validate(arr, w, h):
            val = np.sum(arr[:, 2])
            if val > best_val:
                best_val = val
                best_arr = arr.copy()
                best_wh = (w, h)

    if best_arr is not None and time.time() < deadline - 15.0:
        w, h = best_wh
        vars0 = best_arr.flatten()
        try:
            res = minimize(
                _objective_and_grad,
                vars0,
                args=(N, w, h),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 8000}
            )
            arr = res.x.reshape(N, 3)
            arr[:, 2] = np.maximum(arr[:, 2], 0.0)
            arr = _separate(arr, w, h, iters=500)
            if _validate(arr, w, h):
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best_arr = arr.copy()
        except Exception:
            pass

    if best_arr is None:
        w, h = _aspect_to_wh(1.0)
        best_arr = _make_initial(N, w, h)
        best_arr = _separate(best_arr, w, h, iters=500)

    return best_arr
