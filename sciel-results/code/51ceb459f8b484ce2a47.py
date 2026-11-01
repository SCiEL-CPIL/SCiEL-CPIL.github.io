import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
# ---- Pre-compute triangle indices for speed ----
def _get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _get_triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]
NT = len(_TRI)

# ---- Core geometry ----
def _signed_areas(points):
    pi = points[_I]; pj = points[_J]; pk = points[_K]
    v1 = pj - pi; v2 = pk - pi
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

def _min_triangle_area(points):
    return 0.5 * np.abs(_signed_areas(points)).min()

def _hull_area(points):
    try:
        return scipy.spatial.ConvexHull(points).volume
    except Exception:
        return 0.0

def _score(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

# ---- Smooth surrogate: maximize soft-min triangle area / hull area ----
def _smooth_objective(x, beta):
    pts = x.reshape(-1, 2)
    sa = _signed_areas(pts)
    absa = np.abs(sa) + 1e-12
    # log-sum-exp soft-min of the 2*area = |cross| values
    m = absa.min()
    softmin = m + np.log(np.mean(np.exp(absa - m)))
    ha = _hull_area(pts)
    if ha < 1e-12:
        return -1e6
    return softmin / (2.0 * ha)

def _smooth_neg(x, beta):
    return -_smooth_objective(x, beta)

# ---- Initializers ----
def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _perturbed_polygon(n, base_radius=1.0, noise_scale=0.05, seed=None, rotation=0.0):
    rng = np.random.RandomState(seed)
    return _regular_polygon(n, base_radius, rotation) + rng.uniform(-noise_scale, noise_scale, (n, 2))

def _concentric(n_outer, n_inner, ratio, seed=0, rot_off=0.0):
    rng = np.random.RandomState(seed)
    oa = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2 * np.pi)
    ia = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / max(1, n_inner) + rng.uniform(0, 2 * np.pi) + rot_off
    outer = np.column_stack([np.cos(oa), np.sin(oa)])
    inner = np.column_stack([ratio * np.cos(ia), ratio * np.sin(ia)])
    return np.vstack([outer, inner])

def _random_in_disk(n, seed=None):
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])

def _random_on_ring(n, radius, seed=None):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])

def _build_inits():
    inits = []
    # regular polygon rotations
    for rot in [0.0, np.pi / 14, np.pi / 7, np.pi / 13, np.pi / 3]:
        inits.append(_regular_polygon(N, 1.0, rot))
    # perturbed polygons
    for seed in range(12):
        for ns in [0.03, 0.08, 0.15, 0.25]:
            inits.append(_perturbed_polygon(N, 1.0, ns, seed=seed + 50, rotation=seed * 0.11))
    # concentric circles
    for seed in range(6):
        inits.append(_concentric(7, 7, 0.5, seed=seed))
    for ratio in [0.3, 0.4, 0.6, 0.7]:
        for (no, ni) in [(8, 6), (6, 8), (9, 5), (5, 9)]:
            inits.append(_concentric(no, ni, ratio, seed=int(ratio * 100)))
    # random in disk
    for seed in range(8):
        inits.append(_random_in_disk(N, seed=seed + 7))
    # random on ring
    for seed in range(6):
        inits.append(_random_on_ring(N, 1.0, seed=seed + 21))
    return inits

def construct():
    start = time.time()
    deadline = start + 140.0   # safety margin under the 150s kill
    best_score = 0.0
    best_points = None

    def _try(x0, beta, maxiter, gtol):
        nonlocal best_score, best_points
        if time.time() > deadline:
            return
        try:
            res = scipy.optimize.minimize(
                _smooth_neg, x0, args=(beta,), method='L-BFGS-B',
                options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'maxfun': 4000}
            )
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score = sc
                best_points = opt.copy()
        except Exception:
            pass

    # Build & rank initializations
    raw = _build_inits()
    ranked = []
    for init in raw:
        try:
            p = _normalize(init)
            ranked.append((_score(p), p))
        except Exception:
            continue
    ranked.sort(key=lambda z: -z[0])
    if ranked:
        s0, p0 = ranked[0]
        best_score, best_points = s0, p0.copy()

    # Coarse-to-fine smooth optimization from the best starts
    betas = [12.0, 40.0, 120.0, 400.0, 1200.0]
    for rank, (sc, p) in enumerate(ranked):
        if time.time() > deadline:
            break
        x = p.flatten()
        for bi, beta in enumerate(betas):
            if time.time() > deadline:
                break
            maxiter = 250 if bi < 2 else 120
            _try(x, beta, maxiter=maxiter, gtol=1e-6)
            # chain: warm-start next beta from current best
            if best_points is not None:
                x = best_points.flatten()

    # A few more random restarts if time permits (global diversity)
    rng = np.random.RandomState(12345)
    while time.time() < deadline - 12:
        x0 = (_perturbed_polygon(N, 1.0, 0.15, seed=rng.randint(1, 10**6),
                                 rotation=rng.uniform(0, np.pi))).flatten()
        _try(x0, 40.0, maxiter=150, gtol=1e-6)
        _try(x0, 400.0, maxiter=120, gtol=1e-6)

    # Final high-precision polish from the best point (tight beta, tight tolerance)
    if best_points is not None and time.time() < deadline - 3:
        x = best_points.flatten()
        for beta, mi, gt in [(1500.0, 300, 1e-8), (5000.0, 300, 1e-9)]:
            if time.time() > deadline:
                break
            _try(x, beta, maxiter=mi, gtol=gt)
            if best_points is not None:
                x = best_points.flatten()

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    return best_points
