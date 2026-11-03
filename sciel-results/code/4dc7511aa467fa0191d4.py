import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
EPS = 1e-9


def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)


def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts, dtype=float)


def _normals(beta):
    out = []
    for k in range(6):
        a = beta + math.pi / 6.0 + k * math.pi / 3.0
        out.append((math.cos(a), math.sin(a)))
    return out


def _enclosure_d(pts, cx, cy, beta):
    # max |dot(p-c, n)| over 6 normals
    best = 0.0
    for nx, ny in _normals(beta):
        d = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        m = max(abs(d.max()), abs(d.min()))
        if m > best:
            best = m
    return best


def _overlap_violation(pts, tol=1e-9):
    # min center distance - 2 (unit hexagons, same orientation)
    n = pts.shape[0]
    mind = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = math.hypot(dx, dy)
            if d < mind:
                mind = d
    return max(0.0, 2.0 - mind + tol)


def _soft_overlap(pts, temp=0.05):
    # smooth penalty based on soft-min of center distances
    n = pts.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dists.append(math.hypot(dx, dy))
    dists = np.array(dists)
    m = dists.min()
    shifted = (dists - m) / temp
    shifted = np.clip(shifted, -50, 50)
    softmin = m + temp * np.log(np.sum(np.exp(shifted)))
    return max(0.0, 2.0 - softmin)


def _smooth_objective(vec, beta, temp=0.05):
    # vec: [cx, cy, x1, y1, ..., xn, yn]
    cx, cy = vec[0], vec[1]
    pts = vec[2:].reshape(-1, 2)
    d = _enclosure_d(pts, cx, cy, beta)
    S = (d + APOTHEM) / APOTHEM
    ov = _soft_overlap(pts, temp)
    return S + 100.0 * ov


def _exact_objective(vec, beta):
    cx, cy = vec[0], vec[1]
    pts = vec[2:].reshape(-1, 2)
    d = _enclosure_d(pts, cx, cy, beta)
    S = (d + APOTHEM) / APOTHEM
    ov = _overlap_violation(pts)
    return S + 1000.0 * ov


def _lattice_seeds():
    seeds = []

    # hexagonal ring (center + 6 + 4 outer)
    ring = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (1, 2)]
    seeds.append(_axial_to_cartesian(ring))

    # two rows
    rows = [(i, 0) for i in range(6)] + [(i, 1) for i in range(5)]
    seeds.append(_axial_to_cartesian(rows))

    # triangle-ish
    tri = [(0, 0), (1, 0), (2, 0), (3, 0),
           (1, 1), (2, 1), (3, 1),
           (2, 2), (3, 2),
           (1, -1), (2, -1)]
    seeds.append(_axial_to_cartesian(tri))

    # diamond / cross
    cross = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    seeds.append(_axial_to_cartesian(cross))

    # 3x4 grid
    grid = [(i, j) for i in range(4) for j in range(3)]
    seeds.append(_axial_to_cartesian(grid))

    # 5x3
    grid2 = [(i, j) for i in range(5) for j in range(3)]
    seeds.append(_axial_to_cartesian(grid2))

    # compact blob
    blob = [(0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)]
    seeds.append(_axial_to_cartesian(blob))

    # elongated
    elong = [(i, 0) for i in range(7)] + [(i, 1) for i in range(4)]
    seeds.append(_axial_to_cartesian(elong))

    return seeds


def _random_seed(scale=1.5):
    # random points in a disk
    pts = []
    for _ in range(N):
        r = scale * math.sqrt(np.random.rand())
        a = 2 * math.pi * np.random.rand()
        pts.append([r * math.cos(a), r * math.sin(a)])
    return np.array(pts)


def _optimize_continuous():
    best_S = float('inf')
    best_vec = None
    best_beta = 0.0

    seeds = _lattice_seeds()
    # add a few random seeds
    for _ in range(6):
        seeds.append(_random_seed(1.5))
    for _ in range(4):
        seeds.append(_random_seed(2.2))

    beta_grid = [k * (math.pi / 6) / 12 for k in range(12)]

    for seed in seeds:
        if not _time_ok():
            break
        # center the seed at origin
        c0 = seed.mean(axis=0)
        seed_c = seed - c0

        for beta in beta_grid:
            if not _time_ok():
                break
            vec0 = np.concatenate([[0.0, 0.0], seed_c.ravel()])

            # coarse smooth
            try:
                res = minimize(_smooth_objective, vec0, args=(beta, 0.05),
                               method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
                vec1 = res.x
            except Exception:
                continue

            # refine with smaller temp
            try:
                res = minimize(_smooth_objective, vec1, args=(beta, 0.01),
                               method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
                vec2 = res.x
            except Exception:
                vec2 = vec1

            # exact polish
            try:
                res = minimize(_exact_objective, vec2, args=(beta,),
                               method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
                vec3 = res.x
            except Exception:
                vec3 = vec2

            cx, cy = vec3[0], vec3[1]
            pts = vec3[2:].reshape(-1, 2)
            ov = _overlap_violation(pts)
            if ov > 1e-6:
                continue
            d = _enclosure_d(pts, cx, cy, beta)
            S = (d + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
                best_vec = vec3.copy()
                best_beta = beta

    return best_S, best_vec, best_beta


def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_S, best_vec, best_beta = _optimize_continuous()

    # Fallback if continuous failed
    if best_vec is None or not (best_S < float('inf')):
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (1, 2)]
        pts = _axial_to_cartesian(axial)
        c0 = pts.mean(axis=0)
        best_vec = np.concatenate([[c0[0], c0[1]], pts.ravel()])
        best_beta = 0.0
        best_S = ( _enclosure_d(pts, c0[0], c0[1], 0.0) + APOTHEM) / APOTHEM

    cx, cy = float(best_vec[0]), float(best_vec[1])
    pts = best_vec[2:].reshape(-1, 2)
    angle_deg = math.degrees(best_beta)

    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [cx, cy]
    d = _enclosure_d(pts, cx, cy, best_beta)
    side = float((d + APOTHEM) / APOTHEM)
    rotation = float(angle_deg)

    return inner, center, side, rotation
