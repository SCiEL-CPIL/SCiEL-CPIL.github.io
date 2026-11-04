"""Circle packing in unit square: pack 26 circles, maximize sum of radii.

New strategy family: feasibility-driven sequential radius growth.
- Build a valid packing via largest-empty-circle placement (Voronoi + circle intersections).
- Monotonically grow radii: repeatedly find the largest feasible radius increase per circle
  (dense candidate grid + exact max-radius-at-position routine).
- Final SLSQP polish on the feasible packing.
"""
import numpy as np
import time
from scipy.spatial import Voronoi, QhullError
from scipy.optimize import minimize

N = 26
EPS = 1e-9


def _max_radius_at(x, y, others_xy, others_r):
    """Largest radius for a circle at (x,y) given other fixed circles and walls."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    m = len(others_xy)
    if m:
        d = np.sqrt((others_xy[:, 0] - x) ** 2 + (others_xy[:, 1] - y) ** 2)
        r = min(r, float(np.min(d - others_r)))
    return max(r, 0.0)


def _max_radius_at_vec(x, y, others_xy, others_r):
    """Vectorized max radius at many candidate positions."""
    r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    m = len(others_xy)
    if m:
        dx = x[:, None] - others_xy[None, :, 0]
        dy = y[:, None] - others_xy[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        r = np.minimum(r, np.min(d - others_r[None, :], axis=1))
    return r


def _largest_empty_circle(xy, r, n_cand=140):
    """Find the point maximizing the largest empty circle radius given current circles."""
    n = len(xy)
    cx = xy[:, 0]
    cy = xy[:, 1]

    pts = np.column_stack([cx, cy])
    pts = np.vstack([pts, [[0, 0], [1, 0], [0, 1], [1, 1]]])
    try:
        vor = Voronoi(pts)
    except QhullError:
        vor = None

    cand = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                     [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5]])
    if vor is not None:
        cand = np.vstack([cand, vor.vertices])
    # Circle-circle intersection points
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                continue
            a = (r[i] ** 2 - r[j] ** 2 + d * d) / (2.0 * d)
            h2 = r[i] ** 2 - a * a
            if h2 < 0:
                continue
            h = np.sqrt(max(h2, 0.0))
            xm = cx[i] + a * dx / d
            ym = cy[i] + a * dy / d
            rx = -dy / d
            ry = dx / d
            cand = np.vstack([cand, [xm + h * rx, ym + h * ry], [xm - h * rx, ym - h * ry]])

    # Grid
    g = np.linspace(0.0, 1.0, n_cand)
    gx, gy = np.meshgrid(g, g)
    cand = np.vstack([cand, np.column_stack([gx.ravel(), gy.ravel()])])

    cand = np.clip(cand, 0.0, 1.0)
    # Deduplicate
    cand = np.unique(np.round(cand, 10), axis=0)

    m = len(xy)
    if m:
        dx = cand[:, 0:1] - xy[:, 0][None, :]
        dy = cand[:, 1:2] - xy[:, 1][None, :]
        d = np.sqrt(dx * dx + dy * dy)
        clear = np.min(d - r[None, :], axis=1)
    else:
        clear = np.full(len(cand), 0.5)
    wall = np.minimum(np.minimum(cand[:, 0], 1.0 - cand[:, 0]),
                      np.minimum(cand[:, 1], 1.0 - cand[:, 1]))
    rad = np.minimum(wall, clear)
    idx = int(np.argmax(rad))
    return float(cand[idx, 0]), float(cand[idx, 1]), float(rad[idx])


def _build_valid_packing(deadline):
    """Place N circles one-by-one at the largest empty circle."""
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        if time.time() > deadline:
            break
        if i == 0:
            x, y, rad = 0.5, 0.5, 0.5
        else:
            x, y, rad = _largest_empty_circle(xy[:i], r[:i])
        xy[i, 0] = x
        xy[i, 1] = y
        r[i] = max(rad, 0.0005)
    return xy, r


def _grow_radii(xy, r, deadline, grid=160, max_sweeps=12):
    """Monotone growth: repeatedly find largest feasible radius increase per circle."""
    n = len(xy)
    cur_xy = xy.copy()
    cur_r = r.copy()
    best_total = float(np.sum(cur_r))
    best_xy = cur_xy.copy()
    best_r = cur_r.copy()

    gx = np.linspace(0.0, 1.0, grid)
    gy = np.linspace(0.0, 1.0, grid)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    cand = np.column_stack([GX.ravel(), GY.ravel()])
    # Add current positions as candidates to preserve feasibility
    cand = np.vstack([cand, cur_xy])
    wall_clear = np.minimum(np.minimum(cand[:, 0], 1.0 - cand[:, 0]),
                            np.minimum(cand[:, 1], 1.0 - cand[:, 1]))

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = cur_xy[mask]
            others_r = cur_r[mask]

            clear = _max_radius_at_vec(cand[:, 0], cand[:, 1], others_xy, others_r)
            rad = np.minimum(wall_clear, clear)
            idx = int(np.argmax(rad))
            new_x, new_y = float(cand[idx, 0]), float(cand[idx, 1])
            new_r = float(rad[idx])

            if new_r > cur_r[i] + 1e-9:
                cur_xy[i, 0] = new_x
                cur_xy[i, 1] = new_y
                cur_r[i] = new_r
                improved = True

        total = float(np.sum(cur_r))
        if total > best_total + 1e-12:
            best_total = total
            best_xy = cur_xy.copy()
            best_r = cur_r.copy()
        if not improved:
            break
    return best_xy, best_r


def _slsqp_polish(xy, r, deadline, max_iter=400):
    """SLSQP polish: maximize sum(r) subject to explicit feasibility constraints."""
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    def obj(v):
        return -float(np.sum(v[2 * n:3 * n]))

    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 * n + i] - (1.0 - v[i])})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n + i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 * n + i] - (1.0 - v[n + i])})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[i] - v[j]) ** 2 + (v[n + i] - v[n + j]) ** 2)
                - (v[2 * n + i] + v[2 * n + j])
            })

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0005, 0.49))

    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-10})
    except Exception:
        return xy, r
    if not res.success and not np.isfinite(res.fun):
        return xy, r
    v = res.x
    new_xy = np.column_stack([v[0:n], v[n:2 * n]])
    new_r = v[2 * n:3 * n]
    # Re-clip to feasibility
    for i in range(n):
        ri = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0], new_xy[i, 1], 1.0 - new_xy[i, 1])
        ri = max(ri, 0.0005)
        for j in range(n):
            if j == i:
                continue
            d = np.hypot(new_xy[i, 0] - new_xy[j, 0], new_xy[i, 1] - new_xy[j, 1])
            ri = min(ri, d - new_r[j])
        new_r[i] = max(ri, 0.0005)
    return new_xy, new_r


def _eval_total(xy, r):
    """Return sum of radii if packing is valid, else -inf."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -EPS or x - ri < -EPS or 1.0 - x - ri < -EPS or y - ri < -EPS or 1.0 - y - ri < -EPS:
            return -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
            if d < r[i] + r[j] - EPS:
                return -np.inf
    return float(np.sum(r))


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # Build a valid packing
    xy, r = _build_valid_packing(deadline)
    total = _eval_total(xy, r)
    if not np.isfinite(total):
        # Fallback: small grid
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            xy[i, 0] = (gx + 0.5) / side
            xy[i, 1] = (gy + 0.5) / side
            r[i] = 0.4 / side
        total = _eval_total(xy, r)
        if not np.isfinite(total):
            r[:] = 0.01

    best_xy = xy.copy()
    best_r = r.copy()
    best_total = total

    # Monotone growth
    gxy, gr = _grow_radii(xy, r, deadline, grid=160, max_sweeps=12)
    gtotal = _eval_total(gxy, gr)
    if np.isfinite(gtotal) and gtotal > best_total:
        best_total = gtotal
        best_xy = gxy.copy()
        best_r = gr.copy()

    # SLSQP polish
    if time.time() < deadline - 15:
        pxy, pr = _slsqp_polish(best_xy, best_r, deadline, max_iter=400)
        ptotal = _eval_total(pxy, pr)
        if np.isfinite(ptotal) and ptotal > best_total:
            best_total = ptotal
            best_xy = pxy.copy()
            best_r = pr.copy()

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.0005)
        for j in range(n):
            if j == i:
                continue
            d = np.hypot(x - best_xy[j, 0], y - best_xy[j, 1])
            ri = min(ri, d - best_r[j])
        ri = max(ri, 0.0005)
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        result[i, 2] = ri

    final_xy = result[:, :2]
    final_r = result[:, 2]
    ftotal = _eval_total(final_xy, final_r)
    if not np.isfinite(ftotal):
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    return result
