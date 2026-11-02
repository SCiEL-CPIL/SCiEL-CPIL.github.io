import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

N = 300

_DEADLINE = 150.0   # wall-clock kill limit
_MARGIN = 8.0       # safety margin -> return well before the kill


def _time_left(start):
    return _DEADLINE - _MARGIN - (time.time() - start)


def _compute_true_score(points):
    """Exact packing radius in double precision: min(min pair dist/2, 1 - max|c|)."""
    norms = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(norms)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_pair, min_wall)


def _hex_lattice_init(n_points, rotation=0.0):
    """Hexagonal lattice of points inside the unit circle, nearest-to-centre first."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    if rotation != 0.0:
        c, s_ = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s_], [s_, c]])
        pts = pts @ R.T
    d = np.linalg.norm(pts, axis=1)
    pts = pts[np.argsort(d)[:n_points]]
    return pts


def _ring_init(n_points, jitter=0.0):
    """Deterministic concentric-ring packing; good baseline for circle packing."""
    pts = [[0.0, 0.0]]
    ring = 1
    while len(pts) < n_points:
        count = 6 * ring
        r = 0.5 * ring
        for m in range(count):
            if len(pts) >= n_points:
                break
            a = 2 * np.pi * m / count
            pts.append([r * np.cos(a), r * np.sin(a)])
        ring += 1
    pts = np.array(pts[:n_points], dtype=float)
    if jitter > 0.0:
        pts = pts + np.random.normal(0.0, jitter, size=pts.shape)
    return pts


def _soft_min_obj_and_grad(x, beta):
    """Negative soft-min of (pair distances/2, wall margins) + analytic gradient."""
    C = x.reshape(N, 2)
    D = C[:, None, :] - C[None, :, :]
    L = np.linalg.norm(D, axis=2)
    L_safe = np.where(L > 1e-12, L, 1.0)

    L_wall = np.linalg.norm(C, axis=1)
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)

    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = L_safe[i_idx, j_idx] / 2.0
    x_walls = 1.0 - L_wall_safe
    x_all = np.concatenate([x_pairs_flat, x_walls])

    min_x = np.min(x_all)
    shifted = x_all - min_x
    exp_terms = np.exp(-beta * shifted)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    val = min_x - (1.0 / beta) * np.log(Z)

    grad = np.zeros_like(C)
    P_pairs = P[:len(i_idx)]
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    U = D_flat / L_flat_safe[:, None]
    W = P_pairs * 0.5
    np.add.at(grad, i_idx, W[:, None] * U)
    np.add.at(grad, j_idx, -W[:, None] * U)

    P_walls = P[len(i_idx):]
    U_walls = -C / L_wall_safe[:, None]
    grad += P_walls[:, None] * U_walls
    return -val, -grad


def _optimize_lbfgs(points, beta, max_iter=300, time_limit=10.0, gtol=1e-6):
    """L-BFGS-B on the soft-min surrogate with a time limit."""
    start = time.time()

    def obj_and_grad(x):
        if time.time() - start > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)

    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-12})
        if res.success or res.status in (1, 2, 4):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points


def _force_directed_refine(points, n_iter=400, time_limit=12.0):
    """Force-directed relaxation: repel close pairs, pull inward, clamp to disk."""
    current = points.copy()
    start = time.time()
    for _ in range(n_iter):
        if time.time() - start > time_limit:
            break
        forces = np.zeros_like(current)
        D = current[:, None, :] - current[None, :, :]
        L = np.linalg.norm(D, axis=2)
        np.fill_diagonal(L, np.inf)
        threshold = 0.15
        close_mask = (L < threshold) & (L > 1e-12)
        if np.any(close_mask):
            L_safe = np.where(close_mask, L, 1.0)
            U = D / L_safe[:, :, None]
            F = np.where(close_mask[:, :, None], 1.0 / L_safe[:, :, None], 0.0)
            for i in range(N):
                forces[i] += np.sum(F[i] * U[i], axis=0)
                forces[i] -= np.sum(F[:, i] * U[:, i], axis=0)
        norms = np.linalg.norm(current, axis=1)
        wall_mask = norms > 0.95
        if np.any(wall_mask):
            forces[wall_mask] += 0.5 * (-current[wall_mask] / norms[wall_mask, None])
        current += 0.001 * forces
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            current[mask] = current[mask] * (0.999 / norms[mask])[:, None]
    return current


def _anneal_beta(points, beta_start, beta_end, n_steps, time_budget,
                 max_iter_per_step=120, gtol=1e-6):
    """Smoothly anneal beta (log-spaced) with a time budget across steps."""
    current = points.copy()
    per_step = time_budget / n_steps
    for i in range(n_steps):
        if _time_left(points is not None and None or 0) < 0:
            break
        frac = i / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        current = _optimize_lbfgs(current, beta, max_iter=max_iter_per_step,
                                  time_limit=per_step, gtol=gtol)
    return current


def construct():
    start = time.time()
    np.random.seed(20210617)

    inits = []
    for rot in (0.0, np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 12, 2 * np.pi / 3):
        inits.append(_hex_lattice_init(N, rotation=rot))
    inits.append(_ring_init(N))
    inits.append(_ring_init(N, jitter=0.002))

    best_points = None
    best_score = -np.inf

    def consider(pts):
        nonlocal best_points, best_score
        s = _compute_true_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    for init in inits:
        if _time_left(start) < 25.0:
            break
        current = init.copy()
        # Phase 1: coarse soft-min structure
        current = _optimize_lbfgs(current, 20.0, max_iter=150, time_limit=8.0)
        consider(current)
        # Phase 2: annealed medium->high beta (smooth schedule)
        if _time_left(start) > 25.0:
            current = _anneal_beta(current, 50.0, 800.0, n_steps=4,
                                   time_budget=min(18.0, _time_left(start) - 25.0),
                                   max_iter_per_step=120)
            consider(current)
        # Phase 3: force-directed structural polish
        if _time_left(start) > 20.0:
            current = _force_directed_refine(current, n_iter=300,
                                             time_limit=min(10.0, _time_left(start) - 20.0))
            consider(current)
        # Phase 4: high-beta gradient refinement
        if _time_left(start) > 18.0:
            current = _optimize_lbfgs(current, 2000.0, max_iter=300,
                                      time_limit=min(12.0, _time_left(start) - 18.0))
            consider(current)

    # Second pass: refine the best solution with a smooth anneal + high precision
    if best_points is not None and _time_left(start) > 30.0:
        current = best_points.copy()
        current = _anneal_beta(current, 500.0, 20000.0, n_steps=5,
                               time_budget=min(35.0, _time_left(start) - 30.0),
                               max_iter_per_step=200, gtol=1e-7)
        consider(current)
        if _time_left(start) > 15.0:
            current = _force_directed_refine(current, n_iter=200,
                                             time_limit=min(8.0, _time_left(start) - 15.0))
            consider(current)
        if _time_left(start) > 12.0:
            current = _optimize_lbfgs(current, 40000.0, max_iter=800,
                                      time_limit=min(12.0, _time_left(start) - 12.0),
                                      gtol=1e-8)
            consider(current)

    # Final high-precision polish pass
    if best_points is not None and _time_left(start) > 10.0:
        current = _optimize_lbfgs(best_points, 100000.0, max_iter=1000,
                                  time_limit=min(10.0, _time_left(start) - 10.0),
                                  gtol=1e-9)
        consider(current)

    if best_points is None:
        best_points = _ring_init(N)

    # Guarantee validity: strictly inside the unit circle, finite
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 0.9999
    if np.any(mask):
        best_points[mask] = best_points[mask] * (0.9999 / norms[mask])[:, None]
    best_points = np.where(np.isfinite(best_points), best_points, 0.0)
    return best_points
