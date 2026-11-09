"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3.0)
S3I = 1.0 / S3
AREA_TRI = 0.5 * S3I  # area of the unit equilateral triangle

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)

# Precompute point masks: which triangles involve each point
_point_masks = []
for p in range(N):
    _point_masks.append(((_idx_i == p) | (_idx_j == p) | (_idx_k == p)))


def _to_cartesian(b):
    """b: (N,2) barycentric coords (u on base AB, w on apex C) -> (N,2) cartesian."""
    x = b[:, 0] + 0.5 * b[:, 1]
    y = H * b[:, 1]
    return np.column_stack((x, y))


def _min_area_cart(points):
    p1 = points[_idx_i]; p2 = points[_idx_j]; p3 = points[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]; dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]; dy2 = p3[:, 1] - p1[:, 1]
    return 0.5 * np.min(np.abs(dx1 * dy2 - dy1 * dx2))


def _min_area_b(b):
    return _min_area_cart(_to_cartesian(b))


def _soft_min_b(b, beta):
    """Smooth surrogate for min triangle area (log-sum-exp on negative areas)."""
    pts = _to_cartesian(b)
    p1 = pts[_idx_i]; p2 = pts[_idx_j]; p3 = pts[_idx_k]
    dx1 = p2[:, 0] - p1[:, 0]; dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]; dy2 = p3[:, 1] - p1[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    s = -beta * areas
    m = np.max(s)
    return -1.0 / beta * (m + np.log(np.sum(np.exp(s - m))))


def _obj_soft(b, beta):
    return -_soft_min_b(b, beta)


def _obj_hard(b):
    return -_min_area_b(b)


_BOUNDS = [(1e-4, 1 - 2 * 1e-4)] * (2 * N)


def _init_symmetric():
    """Row-based symmetric layout (base-heavy), returned as barycentric coords."""
    rows = [(0.06, 4), (0.20, 3), (0.34, 3), (0.46, 1)]
    b = np.zeros((N, 2))
    pi = 0
    for w, n in rows:
        u_min = 0.5 * w
        u_max = 1.0 - 0.5 * w
        if n == 1:
            b[pi, 0] = 0.5
        else:
            m = 0.18
            xs = np.linspace(u_min + m * (u_max - u_min),
                             u_max - m * (u_max - u_min), n)
            b[pi:pi + n, 0] = xs
        b[pi:pi + n, 1] = w
        pi += n
    return b


def _init_random(rng):
    b = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random(); w = rng.random()
            if u + w <= 1.0:
                b[i, 0] = u; b[i, 1] = w
                break
    return b


def _init_lattice(rng):
    cands = []
    nd = 5
    for i in range(nd + 1):
        for j in range(nd + 1):
            u = i / nd; w = j / nd
            if u + w <= 1.0 + 1e-9:
                cands.append((u, w))
    cands = np.array(cands)
    sel = [cands[0]]
    for _ in range(N - 1):
        d = np.min([np.linalg.norm(cands - s, axis=1) for s in sel], axis=0)
        sel.append(cands[int(np.argmax(d))])
    b = np.array(sel) + rng.normal(0, 0.02, size=(N, 2))
    return b


def _init_cluster(rng):
    """Base-heavy random cluster (points concentrated near the base)."""
    b = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random(); w = rng.random() ** 1.7
            if u + w <= 1.0:
                b[i, 0] = u; b[i, 1] = w
                break
    return b


def _init_jitter(base, rng, amt):
    return base + rng.normal(0, amt, size=base.shape)


def _de_soft(b, beta, time_left, seed):
    try:
        r = differential_evolution(
            lambda x: _obj_soft(x, beta), _BOUNDS,
            maxiter=18, popsize=20, tol=1e-9, seed=seed,
            polish=False, init='latinhypercube', mutation=(0.5, 1.0), recombination=0.7)
        return r.x.reshape(N, 2)
    except Exception:
        return b


def _sa_anneal(b, time_left, seed):
    """Simulated-annealing restarts with the soft-min objective."""
    rng = np.random.default_rng(seed)
    cur = b.copy(); cur_score = _soft_min_b(cur, 20.0)
    best = cur.copy(); best_score = cur_score
    t = 0.03
    cooling = 0.97
    while time.time() < time_left and t > 1e-5:
        cand = cur + rng.normal(0, t, size=cur.shape)
        cand = np.clip(cand, 1e-4, 1 - 2 * 1e-4)
        cs = _soft_min_b(cand, 20.0)
        if cs >= cur_score or rng.random() < np.exp((cs - cur_score) / max(t, 1e-9)):
            cur = cand; cur_score = cs
            if cur_score > best_score:
                best_score = cur_score; best = cur.copy()
        t *= cooling
    return best


def _nm_polish(b, time_left, seed):
    """Exact-objective Nelder-Mead polish (feasible by construction)."""
    try:
        r = minimize(_obj_hard, b.flatten(), method='Nelder-Mead',
                     options={'maxiter': 6000, 'maxfev': 12000,
                              'xatol': 1e-12, 'fatol': 1e-14})
        return r.x.reshape(N, 2)
    except Exception:
        return b


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before the 150s kill
    rng = np.random.default_rng(2024)

    inits = [
        _init_symmetric(),
        _init_lattice(rng),
        _init_random(rng),
        _init_cluster(rng),
        _init_jitter(_init_symmetric(), rng, 0.03),
        _init_jitter(_init_lattice(rng), rng, 0.05),
        _init_jitter(_init_random(rng), rng, 0.02),
    ]

    best_b = None
    best_val = -np.inf

    for k, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
        seed = 100 + k
        # 1) global soft-min search from this diverse start
        cand = _de_soft(init, 25.0, deadline - 15, seed)
        # 2) simulated-annealing restarts (cheap, good for local basins)
        if time.time() < deadline - 15:
            cand2 = _sa_anneal(cand, deadline - 15, seed + 7)
            if _min_area_b(cand2) > _min_area_b(cand):
                cand = cand2
        # 3) exact polish
        if time.time() < deadline - 8:
            cand = _nm_polish(cand, deadline - 8, seed + 13)
        val = _min_area_b(cand)
        if val > best_val:
            best_val = val
            best_b = cand.copy()

    # 4) final high-precision polish of the overall best
    if best_b is not None and time.time() < deadline - 3:
        polished = _nm_polish(best_b, deadline - 2, 999)
        if _min_area_b(polished) > best_val:
            best_b = polished

    if best_b is None:
        best_b = _init_symmetric()

    pts = _to_cartesian(best_b)
    # clamp to the triangle as a final safety net (should be a no-op)
    for i in range(N):
        x, y = pts[i]
        y = max(0.0, min(H, y))
        xmin = y * S3I; xmax = 1.0 - y * S3I
        if xmin > xmax:
            x, y = 0.5, 0.0
        else:
            x = max(xmin, min(xmax, x))
        pts[i] = [x, y]
    return pts
