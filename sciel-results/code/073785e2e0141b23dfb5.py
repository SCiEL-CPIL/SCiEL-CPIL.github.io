"""
Circle packing in a perimeter-4 rectangle.

Strategy (Voronoi / inradius model):
  * Represent the packing by a set of points (cell centers) in a rectangle
    of perimeter 4 (w + h = 2).
  * Build the Voronoi diagram of the points clipped to the rectangle.
  * The feasible radius of cell i is the inradius of its Voronoi cell
    (distance to the nearest wall / edge / vertex).
  * The inradius is a concave (min-of-convex) function of the center, so
    the sum is concave; we do gradient ascent on it.
  * Multi-start over aspect ratios and initial layouts; a fast grid fallback
    guarantees a valid answer within the deadline.
"""

import numpy as np
import time
import random

N = 21
DEADLINE = 145.0  # safety margin before the 150s kill


# ----------------------------------------------------------------------------
# Voronoi inradius computation (scipy if available, else a robust fallback)
# ----------------------------------------------------------------------------
try:
    from scipy.spatial import Voronoi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _inradii(points, w, h, use_scipy=True):
    """Return the inradius (max feasible radius) for each point's Voronoi cell."""
    n = len(points)
    if _HAVE_SCIPY and use_scipy:
        try:
            v = Voronoi(points)
            r = np.zeros(n)
            for i in range(n):
                pi = points[i]
                # wall distances
                r[i] = min(pi[0], w - pi[0], pi[1], h - pi[1])
                reg = v.regions[v.point_region[i]]
                if -1 in reg:
                    # unbounded cell: the wall distance already bounds it
                    continue
                for j in reg:
                    if j < 0:
                        continue
                    p0 = v.vertices[j]
                    d = np.hypot(pi[0] - p0[0], pi[1] - p0[1])
                    if d < r[i]:
                        r[i] = d
            # numerical safety
            r = np.clip(r, 0.0, None)
            return r
        except Exception:
            pass
    # Fallback: min distance to all other points / 2, and to walls.
    r = np.zeros(n)
    for i in range(n):
        xi, yi = points[i]
        ri = min(xi, w - xi, yi, h - yi)
        for j in range(n):
            if i != j:
                d = np.hypot(xi - points[j][0], yi - points[j][1])
                if d * 0.5 < ri:
                    ri = d * 0.5
        r[i] = ri
    return np.clip(r, 0.0, None)


# ----------------------------------------------------------------------------
# Gradient ascent on the sum of inradii (concave objective)
# ----------------------------------------------------------------------------
def _grad_ascent(points, w, h, budget, use_scipy=True):
    """Local gradient ascent on sum of inradii. Returns (points, score)."""
    points = points.astype(float).copy()
    n = len(points)
    score = float(np.sum(_inradii(points, w, h, use_scipy)))

    lr = 0.02
    eps = 1e-3
    best_points = points.copy()
    best_score = score
    stall = 0

    while budget() > 0.3:
        # central-difference gradient
        grad = np.zeros_like(points)
        for i in range(n):
            for d in range(2):
                p = points.copy()
                p[i, d] += eps
                if p[i, d] < 0 or p[i, d] > (w if d == 0 else h):
                    p[i, d] = points[i, d]
                    continue
                f_plus = float(np.sum(_inradii(p, w, h, use_scipy)))
                p2 = points.copy()
                p2[i, d] -= eps
                if p2[i, d] < 0 or p2[i, d] > (w if d == 0 else h):
                    p2[i, d] = points[i, d]
                    continue
                f_minus = float(np.sum(_inradii(p2, w, h, use_scipy)))
                grad[i, d] = (f_plus - f_minus) / (2.0 * eps)

        # step
        step = points + lr * grad
        step[:, 0] = np.clip(step[:, 0], 0.0, w)
        step[:, 1] = np.clip(step[:, 1], 0.0, h)

        new_score = float(np.sum(_inradii(step, w, h, use_scipy)))
        if new_score > score + 1e-9:
            points = step
            score = new_score
            if score > best_score:
                best_score = score
                best_points = points.copy()
            stall = 0
            lr = min(lr * 1.15, 0.08)
        else:
            stall += 1
            lr *= 0.6
            if lr < 1e-4:
                break

    return best_points, best_score


# ----------------------------------------------------------------------------
# Initial layouts
# ----------------------------------------------------------------------------
def _grid_layout(w, h, n=N):
    cols = int(round(np.sqrt(n * (w / h))))
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    pts = []
    for i in range(n):
        r = i // cols
        c = i % cols
        pts.append([w * (c + 0.5) / cols, h * (r + 0.5) / rows])
    return np.array(pts[:n])


def _hex_layout(w, h, n=N):
    # hexagonal-ish packing, scaled to fill the box
    pts = []
    spacing = 1.0
    row = 0
    while len(pts) < n:
        for col in range(200):
            x = (col + (0.5 if row % 2 else 0.0)) * spacing
            y = row * spacing * 0.866
            if x < w and y < h:
                pts.append([x, y])
            if len(pts) >= n:
                break
        row += 1
    pts = np.array(pts[:n])
    # scale to fill the box
    if pts.shape[0]:
        pts[:, 0] *= w / max(1e-9, pts[:, 0].max())
        pts[:, 1] *= h / max(1e-9, pts[:, 1].max())
        pts[:, 0] = np.clip(pts[:, 0], 0, w)
        pts[:, 1] = np.clip(pts[:, 1], 0, h)
    return pts


def _random_layout(w, h, n=N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n, 2)) * np.array([w, h])


# ----------------------------------------------------------------------------
# Construct
# ----------------------------------------------------------------------------
def construct():
    start = time.time()

    def budget():
        return DEADLINE - time.time()

    best_score = -1.0
    best_points = None
    best_wh = (1.0, 1.0)

    # Candidate aspect ratios (w, h) with w + h = 2
    aspects = [1.0, 0.8, 1.25, 0.7, 1.43, 0.62, 1.6, 0.55, 1.8, 0.5]
    # interleave layouts
    layouts = [_grid_layout, _hex_layout, _random_layout]

    # Phase 1: cheap grid fallback to guarantee a valid answer quickly
    for w, h in [(1.0, 1.0)]:
        pts = _grid_layout(w, h)
        r = _inradii(pts, w, h, use_scipy=False)
        s = float(np.sum(r))
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            best_wh = (w, h)

    if budget() < 5.0:
        return _finalize(best_points, best_wh)

    # Phase 2: multi-start gradient ascent
    seed_counter = 0
    for w in aspects:
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        for li, lay in enumerate(layouts):
            if budget() < 4.0:
                break
            seed_counter += 1
            if lay is _random_layout:
                pts = _random_layout(w, h, seed=seed_counter)
            else:
                pts = lay(w, h)
            # give each start a bounded time
            t0 = time.time()
            per_start = min(12.0, max(2.0, (DEADLINE - 6.0 - (time.time() - start)) / 8.0))
            def local_budget():
                return per_start - (time.time() - t0)
            pts2, s2 = _grad_ascent(pts, w, h, local_budget, use_scipy=_HAVE_SCIPY)
            if s2 > best_score:
                best_score = s2
                best_points = pts2.copy()
                best_wh = (w, h)

    # Phase 3: if time remains, polish the best solution with more ascent
    if best_points is not None and budget() > 8.0:
        w, h = best_wh
        pts2, s2 = _grad_ascent(best_points, w, h, lambda: budget() - 3.0,
                                use_scipy=_HAVE_SCIPY)
        if s2 > best_score:
            best_score = s2
            best_points = pts2.copy()
            best_wh = (w, h)

    return _finalize(best_points, best_wh)


def _finalize(points, wh):
    w, h = wh
    if points is None:
        points = _grid_layout(1.0, 1.0)
        w, h = 1.0, 1.0
    points = np.clip(points, 0.0, np.array([w, h]))
    r = _inradii(points, w, h, use_scipy=_HAVE_SCIPY)
    r = np.clip(r, 0.0, None)
    # tiny shrink to guarantee strictly disjoint interiors
    r = r * 0.999
    out = np.column_stack((points, r))
    return out


if __name__ == "__main__":
    res = construct()
    print("shape", res.shape, "sum r", float(np.sum(res[:, 2])))
