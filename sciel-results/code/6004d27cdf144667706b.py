"""Pack 200 equal circles in unit square [-0.5, 0.5]^2 maximizing common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _hex_init(n):
    """Hexagonal lattice init, scaled to fit square, trimmed to n points."""
    pts = []
    max_col = 16
    max_row = 15
    for j in range(max_row):
        for i in range(max_col):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3.0) / 2.0
            pts.append((x, y))
    pts = np.array(pts, dtype=float)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    scale = min(1.0 / w, 1.0 / h) * 0.92
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    pts = (pts - np.array([cx, cy])) * scale
    if len(pts) > n:
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)[:n]
        pts = pts[idx]
    elif len(pts) < n:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:n]
    return pts.astype(float)


def _compute_radius(c):
    c = np.asarray(c)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    bd = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    r_bound = bd.min()
    if r_bound < 0:
        return 0.0
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = d.min() / 2.0
    return min(r_bound, r_pair)


def _soft_min_value_grad(c, t):
    """f approximates -min(vals); minimize f to maximize min(vals). Upper-triangle pairs."""
    c = np.asarray(c)
    n = len(c)
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt(np.sum(diff * diff, axis=1))
    bd = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    vals = np.concatenate([d, bd])
    vmin = vals.min()
    e = np.exp(-t * (vals - vmin))
    s = e.sum()
    if s <= 0:
        return 0.0, np.zeros_like(c)
    f = vmin - np.log(s) / t
    w = e / s
    wpair = w[:len(d)]
    wbound = w[len(d):]
    grad = np.zeros_like(c)
    # Pairwise gradient
    for k in range(len(d)):
        a = iu[0][k]
        b = iu[1][k]
        wk = wpair[k]
        dk = d[k]
        if dk < 1e-12:
            continue
        v = (c[a] - c[b]) / dk
        grad[a] += wk * v
        grad[b] -= wk * v
    # Boundary gradient
    for i in range(n):
        wb = wbound[i]
        if wb <= 0:
            continue
        dx = 0.5 - np.abs(c[i, 0])
        dy = 0.5 - np.abs(c[i, 1])
        if dx <= dy:
            gx = -np.sign(c[i, 0]) if c[i, 0] != 0 else 0.0
            grad[i, 0] += wb * gx
        else:
            gy = -np.sign(c[i, 1]) if c[i, 1] != 0 else 0.0
            grad[i, 1] += wb * gy
    return f, grad


def _slsqp_step(c, t, maxiter):
    def obj(x):
        f, _ = _soft_min_value_grad(x.reshape(N, 2), t)
        return f
    def obj_grad(x):
        _, g = _soft_min_value_grad(x.reshape(N, 2), t)
        return g.ravel()
    bounds = [(-0.5, 0.5)] * (2 * N)
    res = minimize(obj, c.ravel(), jac=obj_grad, method='SLSQP',
                   bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False})
    return res.x.reshape(N, 2)


def construct():
    t0 = time.time()
    c = _hex_init(N)
    best_c = c.copy()
    best_r = _compute_radius(c)

    temps = [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
    maxiters = [15, 20, 30, 40, 50, 60, 80, 100]

    for t, mi in zip(temps, maxiters):
        if time.time() - t0 > DEADLINE - 15:
            break
        try:
            c_new = _slsqp_step(c, t, mi)
            if np.all(np.isfinite(c_new)):
                r_new = _compute_radius(c_new)
                if r_new >= best_r - 1e-12:
                    c = c_new
                    best_c = c_new.copy()
                    best_r = r_new
        except Exception:
            pass

    if time.time() - t0 < DEADLINE - 10:
        try:
            c_new = _slsqp_step(c, 0.002, 150)
            if np.all(np.isfinite(c_new)):
                r_new = _compute_radius(c_new)
                if r_new >= best_r - 1e-12:
                    best_c = c_new.copy()
                    best_r = r_new
        except Exception:
            pass

    return best_c


if __name__ == '__main__':
    c = construct()
    print("radius:", _compute_radius(c))
