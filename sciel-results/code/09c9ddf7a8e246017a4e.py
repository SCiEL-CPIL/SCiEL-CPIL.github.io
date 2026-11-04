"""Circle packing in unit square: 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _eval_packing(xy, r):
    """Returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _logsumexp(v, t):
    """Numerically stable log-sum-exp of vector v with temperature t."""
    m = np.max(v)
    if m > 1e15:
        return m
    return m + t * np.log(np.sum(np.exp((v - m) / t)))


def _barrier_objective(p, n, t, w):
    """
    Objective = -sum(r) + w * (sum soft wall violations + sum soft overlap violations)
    Soft violations use log-sum-exp -> smooth, tight as t->0.
    """
    xy = p[0::3].reshape(-1, 2)
    r = np.maximum(p[1::3], 1e-4)

    # Wall violations per circle: max(ri - x, ri - (1-x), ri - y, ri - (1-y), 0)
    wall = np.empty(n)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        vals = np.array([ri - x, ri - (1.0 - x), ri - y, ri - (1.0 - y), 0.0])
        wall[i] = _logsumexp(vals, t)

    # Overlap violations per pair: max(ri+rj - dist, 0)
    overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            vals = np.array([r[i] + r[j] - dist, 0.0])
            overlap += _logsumexp(vals, t)

    return -float(np.sum(r)) + w * (float(np.sum(wall)) + overlap)


def _hex_init():
    """Hexagonal lattice packing of N circles, normalized to fit in [0,1]^2."""
    pts = []
    # Use a hex lattice with enough points, then take first N
    side = int(math.ceil(math.sqrt(N)))
    row_h = math.sqrt(3) / 2.0
    # generate a grid of hex points
    for row in range(side + 2):
        for col in range(side + 2):
            x = col * 1.0 + (0.5 if row % 2 else 0.0)
            y = row * row_h
            pts.append([x, y])
    pts = np.array(pts[:N])
    # Normalize to fit in [0,1]^2 with margin
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    w = xmax - xmin
    h = ymax - ymin
    scale = 1.0 / max(w, h)
    pts = (pts - np.array([xmin, ymin])) * scale
    # Center
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    pts = pts - np.array([cx - 0.5, cy - 0.5])
    # Initial radii: small
    r = np.full(N, 0.03)
    p = np.zeros(3 * N)
    p[0::3] = pts[:, 0]
    p[1::3] = pts[:, 1]
    p[2::3] = r
    return p


def _grid_init():
    """Square grid init."""
    side = int(math.ceil(math.sqrt(N)))
    pts = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        gx = i % side
        gy = i // side
        pts[i, 0] = (gx + 0.5) / side
        pts[i, 1] = (gy + 0.5) / side
        r[i] = 0.5 / side - 0.002
    p = np.zeros(3 * N)
    p[0::3] = pts[:, 0]
    p[1::3] = pts[:, 1]
    p[2::3] = r
    return p


def _slsqp_optimize(p0, deadline, t_start=0.05, t_min=0.001, w_start=500.0, max_iter=200):
    """
    Iteratively sharpen the barrier (decrease t, increase w) and run SLSQP.
    Returns best valid (xy, r, total).
    """
    n = N
    best_xy = p0[0::3].reshape(-1, 2).copy()
    best_r = np.maximum(p0[1::3], 1e-4).copy()
    best_total, best_valid = _eval_packing(best_xy, best_r)
    if not best_valid:
        best_total = 0.0

    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.9999))  # x
        bounds.append((1e-4, 0.9999))  # y
        bounds.append((1e-4, 0.45))    # r

    t = t_start
    w = w_start
    cur = p0.copy()
    # Sharpening schedule
    steps = 6
    for s in range(steps):
        if time.time() > deadline:
            break
        res = minimize(
            _barrier_objective,
            cur,
            args=(n, t, w),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        cur = res.x
        opt_xy = cur[0::3].reshape(-1, 2)
        opt_r = np.maximum(cur[1::3], 1e-4)
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = opt_xy.copy()
            best_r = opt_r.copy()
            cur = np.concatenate([best_xy.ravel(), best_r])
        # Sharpen
        t = max(t_min, t * 0.3)
        w = w * 3.0

    return best_xy, best_r, best_total, best_valid


def _best_fit_local_search(xy, r, deadline, grid_size=50, max_sweeps=15):
    """Best-fit local search: for each circle, find best position+radius on a grid."""
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]
            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        if not improved:
            break
    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Try multiple initializations
    inits = [_hex_init(), _grid_init()]
    # Add a slightly perturbed hex
    p_hex = _hex_init()
    rng = np.random.default_rng(42)
    p_hex_pert = p_hex.copy()
    p_hex_pert[0::3] += rng.uniform(-0.02, 0.02, N)
    p_hex_pert[1::3] += rng.uniform(-0.02, 0.02, N)
    inits.append(p_hex_pert)

    for p0 in inits:
        if time.time() > deadline - 20.0:
            break
        xy, r, total, valid = _slsqp_optimize(p0, deadline - 20.0)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r

    # Fallback if nothing valid
    if best_xy is None or not best_valid:
        p0 = _grid_init()
        xy, r, total, valid = _eval_packing(p0[0::3].reshape(-1, 2), np.maximum(p0[1::3], 1e-4))
        if valid:
            best_total = total
            best_valid = True
            best_xy = p0[0::3].reshape(-1, 2)
            best_r = np.maximum(p0[1::3], 1e-4)
        else:
            side = int(math.ceil(math.sqrt(N)))
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Best-fit local search polish
    if time.time() < deadline - 15.0:
        lf_xy, lf_r, lf_total, lf_valid = _best_fit_local_search(
            best_xy.copy(), best_r.copy(), deadline - 15.0, grid_size=50, max_sweeps=12)
        if lf_valid and lf_total > best_total:
            best_total = lf_total
            best_valid = True
            best_xy = lf_xy
            best_r = lf_r

    # Final SLSQP re-polish from the best-fit result
    if time.time() < deadline - 5.0:
        p_cur = np.concatenate([best_xy.ravel(), best_r])
        fp_xy, fp_r, fp_total, fp_valid = _slsqp_optimize(
            p_cur, deadline - 3.0, t_start=0.02, t_min=0.001, w_start=1000.0, max_iter=150)
        if fp_valid and fp_total > best_total:
            best_total = fp_total
            best_valid = True
            best_xy = fp_xy
            best_r = fp_r

    # Build result with clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 1e-4)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final validation
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(math.ceil(math.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
