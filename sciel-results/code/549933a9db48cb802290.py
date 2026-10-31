"""Circle packing in perimeter-4 rectangle.
Approach: Apollonius / Laguerre-Voronoi packing.
- The max radius a circle can have at a fixed position equals the min distance to
  the four walls and to the *centers* of the other circles (weighted by their radii),
  which is exactly the inscribed-circle radius of its Voronoi (Apollonius) cell.
- We grow circles to that exact radius, then push overlapping circles toward the
  centroid of their free Voronoi region (the "cell center"), then re-grow. Iterating
  converges to a tight, gap-free packing. No penalty terms, no SLSQP.
"""
import numpy as np
import time

N = 21
DEADLINE_PAD = 20.0


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_PAD

    best = None
    best_val = 0.0

    # Several aspect ratios for diversity (w + h = 2).
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55,
                     1.25, 0.8, 1.4, 0.7, 1.6, 0.63, 1.1, 0.9]

    for w0 in aspect_ratios:
        if time.time() > deadline - 15:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue

        x, y = _grid_init(w0, h0)
        r = np.full(N, min(w0, h0) / 5.0 * 0.5)

        x, y, r = _apollonius_pack(x, y, r, w0, h0, deadline)

        val = _sum_if_valid(x, y, r, w0, h0)
        if val > best_val:
            best_val = val
            best = (x.copy(), y.copy(), r.copy(), w0, h0)

    if best is None:
        # Fallback: trivial valid grid in a unit square.
        w0, h0 = 1.0, 1.0
        x, y = _grid_init(w0, h0)
        r = np.full(N, 0.05)
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)
        return np.column_stack((x, y, r))

    x, y, r, w0, h0 = best
    # Final safety shrink to guarantee strict validity.
    for _ in range(20):
        if _is_valid(x, y, r * (1 - 1e-7), w0, h0):
            r = r * (1 - 1e-7)
        else:
            break

    return np.column_stack((x, y, r))


def _grid_init(w, h):
    side = int(np.ceil(np.sqrt(N)))
    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
    return x, y


def _max_radius(i, x, y, r, w, h):
    """Exact max radius of circle i at its current position (Apollonius)."""
    # Wall distances.
    mr = min(x[i], w - x[i], y[i], h - y[i])
    # Neighbor constraints: need d >= r_i + r_j  =>  r_i <= d - r_j.
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.hypot(dx, dy)
        cand = d - r[j]
        if cand < mr:
            mr = cand
    return max(0.0, mr)


def _cell_center(i, x, y, r, w, h):
    """Centroid of the free (unoccupied) region of circle i's Voronoi cell.
    Approximate by sampling the rectangle and keeping points closer (weighted)
    to i than to any other circle, and inside the walls."""
    # Bounding box of the cell: within walls and within r_i + max_neighbor of center.
    lo_x = max(0.0, x[i] - (r[i] + 0.3))
    hi_x = min(w, x[i] + (r[i] + 0.3))
    lo_y = max(0.0, y[i] - (r[i] + 0.3))
    hi_y = min(h, y[i] + (r[i] + 0.3))
    if hi_x <= lo_x or hi_y <= lo_y:
        return x[i], y[i]

    sx, sy = np.linspace(lo_x, hi_x, 12)
    sy_grid, sx_grid = np.meshgrid(sy, sx)
    pts = np.stack([sx_grid.ravel(), sy_grid.ravel()], axis=1)

    # Weighted distance to i: d - r_i.  To j: d - r_j.
    di = np.sqrt((pts[:, 0] - x[i]) ** 2 + (pts[:, 1] - y[i]) ** 2) - r[i]
    mask = np.ones(len(pts), dtype=bool)
    for j in range(N):
        if i == j:
            continue
        dj = np.sqrt((pts[:, 0] - x[j]) ** 2 + (pts[:, 1] - y[j]) ** 2) - r[j]
        mask &= (di < dj)
    pts = pts[mask]
    if len(pts) == 0:
        return x[i], y[i]
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _apollonius_pack(x, y, r, w, h, deadline):
    """Iteratively grow circles to their exact Apollonius radius and push
    overlapping circles toward their cell centers. Returns improved arrays."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)

    # Phase A: grow to exact radius (positions fixed).
    for _ in range(60):
        if time.time() > deadline:
            break
        changed = False
        for i in range(N):
            mr = _max_radius(i, x, y, r, w, h)
            if mr > r[i] + 1e-9:
                r[i] = mr
                changed = True
        if not changed:
            break

    # Phase B: push overlapping circles toward cell centers, then re-grow.
    for outer in range(400):
        if time.time() > deadline - 5:
            break

        # Find all overlapping pairs.
        overlaps = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j] - 1e-10:
                    overlaps.append((i, j, r[i] + r[j] - d))
        if not overlaps:
            break

        # Compute net push direction per circle: toward cell center, weighted.
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i, j, ov in overlaps:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < 1e-9:
                dx, dy, d = 1e-4, 0.0, 1e-4
            ux, uy = dx / d, dy / d
            # Push i away from j and j away from i, scaled by overlap.
            fx[i] += ux * ov
            fy[i] += uy * ov
            fx[j] -= ux * ov
            fy[j] -= uy * ov

        # Add a gentle pull toward each circle's own cell center (helps find
        # the right "pocket" and reduces jitter).
        for i in range(N):
            cx, cy = _cell_center(i, x, y, r, w, h)
            fx[i] += 0.15 * (cx - x[i])
            fy[i] += 0.15 * (cy - y[i])

        # Adaptive step.
        mag = np.max(np.hypot(fx, fy)) if N else 0.0
        step = 0.5 / (1.0 + mag)
        step = min(step, 0.02)

        for i in range(N):
            x[i] += step * fx[i]
            y[i] += step * fy[i]
            # Keep inside walls.
            x[i] = min(max(x[i], r[i]), w - r[i])
            y[i] = min(max(y[i], r[i]), h - r[i])

        # Re-grow to exact radius.
        for i in range(N):
            mr = _max_radius(i, x, y, r, w, h)
            if mr > r[i] + 1e-9:
                r[i] = mr

    # Phase C: final local expansion sweep (binary search per circle).
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            mr = _max_radius(i, x, y, r, w, h)
            if mr > r[i] + 1e-8:
                r[i] = mr
                improved = True
        if not improved:
            break

    return x, y, r


def _is_valid(x, y, r, w, h, tol=1e-8):
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > w + tol) or np.any(y + r > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _sum_if_valid(x, y, r, w, h):
    if _is_valid(x, y, r, w, h):
        return float(np.sum(r))
    return 0.0
