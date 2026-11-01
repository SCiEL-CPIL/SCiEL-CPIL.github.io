"""Combined program: hex seed + physics grow + L-BFGS-B refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32
_EPS = 1e-9


def _validity_check(arr, n=N, eps=_EPS):
    if arr is None or len(arr) != n:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - eps:
                return False
    return True


def _score(arr):
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf


def _hex_seed(n=N):
    """Staggered hexagonal lattice trimmed to n circles, with a base radius."""
    row_h = np.sqrt(3) / 2
    centers = []
    cols = 0
    while len(centers) < n + 8:
        y = row_h * cols
        xoff = 0.5 if (cols % 2) else 0.0
        for i in range(20):
            centers.append((xoff + i, y))
            if len(centers) >= n + 8:
                break
        cols += 1
    centers = centers[:n]
    cx = np.array([c[0] for c in centers], dtype=float)
    cy = np.array([c[1] for c in centers], dtype=float)
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    r = np.full(n, 0.0)
    for i in range(n):
        r[i] = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i]) * 0.5
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])


def _shrink_to_valid(arr, n=N):
    """Shrink radii minimally to guarantee validity (inside + disjoint)."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    for i in range(n):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
    for _ in range(300):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s - _EPS:
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])


def _objective(params, n=N):
    x = params[:n]; y = params[n:2*n]; r = params[2*n:3*n]
    penalty = 0.0
    for i in range(n):
        if x[i] - r[i] < 0: penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1: penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0: penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1: penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0: penalty += r[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            s = r[i] + r[j]
            if d2 < s ** 2:
                penalty += (s ** 2 - d2)
    return -np.sum(r) + 100.0 * penalty


def _lbfgs_refine(circles, n=N, maxiter=300):
    """Local L-BFGS-B refinement from a given configuration."""
    params = circles.flatten().copy()
    try:
        result = minimize(_objective, params, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
        new = result.x.reshape(n, 3)
    except Exception:
        return circles
    new = _shrink_to_valid(new, n)
    return new


def _repulsion_optimize(x0, y0, r0, n=N, max_iter=2500, dt=0.02, grow=0.0008,
                        wall_k=40.0, pair_k=60.0, max_r=0.25, seed=0):
    """Physics-style repel + grow; returns best valid config found."""
    rng = np.random.default_rng(seed)
    x = x0.copy().astype(float); y = y0.copy().astype(float)
    r = np.minimum(r0.copy().astype(float), max_r)
    best = None; best_score = -np.inf

    def total_penalty():
        pen = 0.0
        for i in range(n):
            if x[i] - r[i] < 0: pen += wall_k * (0 - (x[i] - r[i])) ** 2
            if x[i] + r[i] > 1: pen += wall_k * ((x[i] + r[i]) - 1) ** 2
            if y[i] - r[i] < 0: pen += wall_k * (0 - (y[i] - r[i])) ** 2
            if y[i] + r[i] > 1: pen += wall_k * ((y[i] + r[i]) - 1) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    pen += pair_k * (s - d) ** 2
        return pen

    for it in range(max_iter):
        fx = np.zeros(n); fy = np.zeros(n)
        for i in range(n):
            if x[i] - r[i] < 0: fx[i] += wall_k * (0 - (x[i] - r[i]))
            if x[i] + r[i] > 1: fx[i] -= wall_k * ((x[i] + r[i]) - 1)
            if y[i] - r[i] < 0: fy[i] += wall_k * (0 - (y[i] - r[i]))
            if y[i] + r[i] > 1: fy[i] -= wall_k * ((y[i] + r[i]) - 1)
            for j in range(i + 1, n):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                d = np.hypot(dx, dy) + 1e-12
                s = r[i] + r[j]
                if d < s:
                    f = pair_k * (s - d) / d
                    fx[i] += f * dx; fy[i] += f * dy
                    fx[j] -= f * dx; fy[j] -= f * dy
        x += dt * fx
        y += dt * fy
        r = np.minimum(r + grow, max_r)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        if it % 20 == 0 or it == max_iter - 1:
            arr = _shrink_to_valid(np.column_stack([x, y, r]), n)
            sc = _score(arr)
            if sc > best_score:
                best_score = sc
                best = arr.copy()
    if best is None:
        best = _shrink_to_valid(np.column_stack([x, y, r]), n)
    return best, best_score


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n = N
    base = _hex_seed(n)
    base = _shrink_to_valid(base, n)
    best = base
    best_score = _score(best)

    rng = np.random.default_rng(12345)
    restart = 0
    while time.time() < deadline - 8:
        x0 = base[:, 0].copy(); y0 = base[:, 1].copy(); r0 = base[:, 2].copy()
        jit = 0.05
        x0 += rng.uniform(-jit, jit, n)
        y0 += rng.uniform(-jit, jit, n)
        x0 = np.clip(x0, 0.0, 1.0)
        y0 = np.clip(y0, 0.0, 1.0)
        r0 = r0 * rng.uniform(0.7, 1.3)
        if restart % 5 == 0:
            perm = rng.permutation(n)
            x0 = x0[perm]; y0 = y0[perm]; r0 = r0[perm]

        # physics grow
        arr, sc = _repulsion_optimize(x0, y0, r0, n=n,
                                      max_iter=2500, dt=0.02, grow=0.0008,
                                      wall_k=40.0, pair_k=60.0, max_r=0.25,
                                      seed=restart)
        # L-BFGS-B local refinement (refinement machinery from current program)
        arr = _lbfgs_refine(arr, n=n, maxiter=300)
        sc = _score(arr)
        if sc > best_score:
            best_score = sc
            best = arr.copy()
        restart += 1

    if not _validity_check(best, n):
        best = _shrink_to_valid(_hex_seed(n), n)
    return best
