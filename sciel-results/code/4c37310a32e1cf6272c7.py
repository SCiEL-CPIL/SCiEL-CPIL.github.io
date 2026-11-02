import math
import numpy as np

N = 12
SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi
DEADLINE = 150.0
MARGIN = 8.0
EPS = 1e-9


def _hex_vertices(cx, cy, side, rot_deg):
    """Vertices of a regular hexagon (circumradius = side) for validation."""
    a = math.radians(rot_deg)
    return [(cx + side * math.cos(a + k * TWO_PI / 6.0),
             cy + side * math.sin(a + k * TWO_PI / 6.0)) for k in range(6)]


def _seg_seg_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > EPS and d2 > EPS) or (d1 < -EPS and d2 < -EPS)):
        return False
    if ((d3 > EPS and d4 > EPS) or (d3 < -EPS and d4 < -EPS)):
        return False
    return True


def _poly_overlap(v1, v2):
    n = len(v1)
    m = len(v2)
    for i in range(n):
        for j in range(m):
            if _seg_seg_intersect(v1[i], v1[(i + 1) % n], v2[j], v2[(j + 1) % m]):
                return True
    return False


def _validate(inner, center, side, angle_deg):
    ox, oy = center
    a = math.radians(angle_deg)
    outer = [(ox + side * math.cos(a + k * TWO_PI / 6.0),
              oy + side * math.sin(a + k * TWO_PI / 6.0)) for k in range(6)]
    for (x, y, ang) in inner:
        v = _hex_vertices(x, y, 1.0, ang)
        if _poly_overlap(v, outer):
            return False
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if _poly_overlap(_hex_vertices(*inner[i]), _hex_vertices(*inner[j])):
                return False
    return True


def _softmin_loss(pos, S, temp, floor):
    """Smooth surrogate for max-min distance with a disk containment constraint."""
    m = pos.shape[0]
    d2 = np.empty((m, m))
    for i in range(m):
        d2[i, i] = 1e30
        d2[i, i + 1:] = np.sum((pos[i] - pos[i + 1:]) ** 2, axis=1)
    d2 = 0.5 * (d2 + d2.T)
    np.fill_diagonal(d2, 1e30)
    d2 = d2 - floor * floor
    t = max(temp, 1e-4)
    row = d2 / t
    row -= row.max(axis=1, keepdims=True)
    soft = t * np.log(np.sum(np.exp(row), axis=1))
    r2 = np.sum(pos * pos, axis=1)
    R = (S - 1.0) * 0.5 * SQRT3
    c = r2 - R * R
    c = c - t
    c -= c.max()
    soft_c = t * np.log(np.sum(np.exp(c / t)))
    return float(np.sum(soft) + np.sum(soft_c))


def _exact_min_dist(pos):
    m = pos.shape[0]
    best = float('inf')
    for i in range(m):
        for j in range(i + 1, m):
            d = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < best:
                best = d
    return best


def _max_radius(pos):
    return float(np.max(np.hypot(pos[:, 0], pos[:, 1])))


def _polish(pos, iters=300):
    """Exact local solve: minimize the outer side length over positions."""
    pos = pos.copy()
    m = pos.shape[0]
    best_side = _max_radius(pos) / (0.5 * SQRT3) + 1.0
    step = 0.05
    for _ in range(iters):
        S = _max_radius(pos) / (0.5 * SQRT3) + 1.0
        best_side = min(best_side, S)
        grad = np.zeros_like(pos)
        for i in range(m):
            r2 = pos[i, 0] ** 2 + pos[i, 1] ** 2
            r = math.sqrt(r2) + 1e-12
            grad[i] = pos[i] / r
        for i in range(m):
            for j in range(i + 1, m):
                diff = pos[i] - pos[j]
                d = math.hypot(diff[0], diff[1])
                if d < 2.0 - 1e-9:
                    f = (2.0 - d) / d
                    grad[i] += f * diff
                    grad[j] -= f * diff
        step *= 0.98
        pos -= step * grad
        if step < 1e-8:
            break
    return pos, best_side


def construct():
    import time
    t0 = time.time()

    # Best feasible (or best objective) solution found so far.
    best_pos = None
    best_side = float('inf')

    def _update(pos):
        nonlocal best_pos, best_side
        S = _max_radius(pos) / (0.5 * SQRT3) + 1.0
        if S < best_side:
            best_side = S
            best_pos = pos.copy()

    # ---- Phase 1: smooth surrogate optimization from several seeds ----
    rng = np.random.default_rng(1234)
    n_seeds = 8
    for s in range(n_seeds):
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        if s == 0:
            # seed 0: hexagonal lattice
            pos = np.array([
                (1.0 * (i - 3) + (0.5 if j % 2 else 0.0) * 0, 0.0)
                for i in range(4)
            ])
            pos = np.array([
                [0.0, 0.0],
                [2.0, 0.0], [-2.0, 0.0],
                [1.0, SQRT3], [-1.0, SQRT3], [0.0, 2 * SQRT3],
                [1.0, -SQRT3], [-1.0, -SQRT3], [0.0, -2 * SQRT3],
                [3.0, 0.0], [-3.0, 0.0], [0.0, 3 * SQRT3],
            ])
        else:
            ang = rng.uniform(0, TWO_PI, N)
            rad = rng.uniform(0.0, 4.0, N)
            pos = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])

        temp = 0.6
        for it in range(400):
            if time.time() - t0 > DEADLINE - MARGIN:
                break
            # finite-difference gradient of the smooth loss
            g = np.zeros_like(pos)
            h = 0.01
            base = _softmin_loss(pos, 4.0, temp, 2.0)
            for i in range(N):
                for d in range(2):
                    pos[i, d] += h
                    lp = _softmin_loss(pos, 4.0, temp, 2.0)
                    pos[i, d] -= 2 * h
                    lm = _softmin_loss(pos, 4.0, temp, 2.0)
                    pos[i, d] += h
                    g[i, d] = (lp - lm) / (2 * h)
            ng = np.linalg.norm(g)
            if ng > 1e-10:
                pos -= 0.02 * g / ng * 0.5  # cautious step
            temp *= 0.97
            if temp < 0.02:
                temp = 0.02
            _update(pos)

    # ---- Phase 2: exact local polish from the best seed ----
    if best_pos is None:
        best_pos = np.zeros((N, 2))
    if time.time() - t0 <= DEADLINE - MARGIN:
        polished, _ = _polish(best_pos, iters=400)
        _update(polished)

    # ---- Build the result ----
    pos = best_pos if best_pos is not None else np.zeros((N, 2))
    S = _max_radius(pos) / (0.5 * SQRT3) + 1.0

    inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
    result = (inner, (0.0, 0.0), float(S), 0.0)

    # ---- Final validation; fall back to a guaranteed-feasible layout if needed ----
    if time.time() - t0 <= DEADLINE - 2.0:
        if not _validate(*result):
            # Guaranteed feasible: unit hexagons on a hex lattice, outer side 6
            pos = np.array([
                [0.0, 0.0],
                [2.0, 0.0], [-2.0, 0.0],
                [1.0, SQRT3], [-1.0, SQRT3], [0.0, 2 * SQRT3],
                [1.0, -SQRT3], [-1.0, -SQRT3], [0.0, -2 * SQRT3],
                [3.0, 0.0], [-3.0, 0.0], [0.0, 3 * SQRT3],
            ])
            S = _max_radius(pos) / (0.5 * SQRT3) + 1.0
            inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
            result = (inner, (0.0, 0.0), float(S), 0.0)
    return result
