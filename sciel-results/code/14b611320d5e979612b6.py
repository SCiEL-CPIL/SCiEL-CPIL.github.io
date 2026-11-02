"""Circle packing in unit square: force-directed + gradient optimization."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26
DEADLINE = 115.0  # safety margin under 120s limit
_T0 = time.time()


def _remaining():
    return DEADLINE - (time.time() - _T0)


def _valid(arr):
    if arr is None or arr.shape != (N, 3):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0) or np.any(x < 0) or np.any(y < 0):
        return False
    if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < (r[i] + r[j]) - 1e-9:
                return False
    return True


def _score(arr):
    return float(arr[:, 2].sum())


def _walls(arr):
    """Wall penetration: x-r<0, y-r<0, x+r>1, y+r>1."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    return np.array([
        np.maximum(0.0, -x + r),
        np.maximum(0.0, -y + r),
        np.maximum(0.0, x + r - 1.0),
        np.maximum(0.0, y + r - 1.0),
    ])


def _contact_pairs(arr):
    """Pairs of circles in contact (distance <= r_i+r_j + tol) via Delaunay."""
    pts = arr[:, :2]
    r = arr[:, 2]
    pairs = []
    try:
        tri = Delaunay(pts)
        for s in tri.simplices:
            for a in range(3):
                i, j = s[a], s[(a + 1) % 3]
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d <= r[i] + r[j] + 1e-7:
                    pairs.append((i, j))
    except Exception:
        # fallback: all pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d <= r[i] + r[j] + 1e-7:
                    pairs.append((i, j))
    return pairs


def _objective(v, pairs):
    """Penalty: maximize sum r minus overlap & wall violations."""
    n = N
    x = v[:2 * n]
    y = v[2 * n:4 * n]
    r = v[4 * n:5 * n]
    # sum radii (to maximize -> minimize negative)
    obj = -np.sum(r)
    # wall violations
    wx = np.maximum(0.0, -x + r)
    wy = np.maximum(0.0, -y + r)
    wxp = np.maximum(0.0, x + r - 1.0)
    wyp = np.maximum(0.0, y + r - 1.0)
    obj += 10.0 * (np.sum(wx) + np.sum(wy) + np.sum(wxp) + np.sum(wyp))
    # overlap violations
    for i, j in pairs:
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        viol = np.maximum(0.0, (r[i] + r[j]) - d)
        obj += 10.0 * viol
    return obj


def _optimize_from(v0, pairs, max_iter=2000):
    """Local gradient optimization from v0."""
    n = N
    v0 = v0.copy()
    # bounds
    lb = np.zeros(5 * n)
    ub = np.zeros(5 * n)
    ub[:2 * n] = 1.0
    ub[2 * n:4 * n] = 1.0
    ub[4 * n:5 * n] = 0.5
    lb[4 * n:5 * n] = 1e-5
    # radius bounds per circle
    for i in range(n):
        ub[4 * n + i] = 0.5

    res = minimize(
        _objective, v0, args=(pairs,),
        method='L-BFGS-B', bounds=list(zip(lb, ub)),
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9,
                 'maxls': 50, 'maxfun': 5000},
    )
    return res.x


def _to_arr(v):
    n = N
    return np.column_stack([v[:2 * n], v[2 * n:4 * n], v[4 * n:5 * n]])


def _grid_initial():
    """Grid of equal circles as a valid starting point."""
    side = int(np.ceil(np.sqrt(N)))
    rad = 0.5 / side
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([(i + 0.5) / side, (j + 0.5) / side, rad])
    arr = np.array(cells[:N])
    return arr


def construct():
    best_arr = None
    best_score = -1.0

    # --- Phase 1: grid seed + optimize ---
    if _remaining() > 20:
        arr0 = _grid_initial()
        if _valid(arr0):
            best_arr = arr0.copy()
            best_score = _score(arr0)
        v0 = arr0.reshape(-1)
        pairs = _contact_pairs(arr0)
        v1 = _optimize_from(v0, pairs)
        arr1 = _to_arr(v1)
        if _valid(arr1) and _score(arr1) > best_score:
            best_score = _score(arr1)
            best_arr = arr1.copy()
        # recompute contacts and polish
        if _remaining() > 15:
            pairs2 = _contact_pairs(arr1)
            v2 = _optimize_from(v1, pairs2, max_iter=3000)
            arr2 = _to_arr(v2)
            if _valid(arr2) and _score(arr2) > best_score:
                best_score = _score(arr2)
                best_arr = arr2.copy()

    # --- Phase 2: perturbation restarts (keep best) ---
    rng = np.random.default_rng(12345)
    base = best_arr if best_arr is not None else _grid_initial()
    for it in range(200):
        if _remaining() < 12:
            break
        # perturb positions slightly
        scale = 0.05 / (1.0 + it * 0.02)
        v = base.reshape(-1).copy()
        v[:2 * N] += rng.normal(0, scale, size=2 * N)
        v[2 * N:4 * N] += rng.normal(0, scale, size=2 * N)
        # keep radii
        pairs = _contact_pairs(_to_arr(v))
        v1 = _optimize_from(v, pairs, max_iter=1500)
        arr1 = _to_arr(v1)
        if _valid(arr1) and _score(arr1) > best_score:
            best_score = _score(arr1)
            best_arr = arr1.copy()

    # --- Phase 3: final high-precision polish ---
    if best_arr is not None and _remaining() > 8:
        pairs = _contact_pairs(best_arr)
        v = best_arr.reshape(-1)
        v1 = _optimize_from(v, pairs, max_iter=5000)
        arr1 = _to_arr(v1)
        if _valid(arr1) and _score(arr1) > best_score:
            best_score = _score(arr1)
            best_arr = arr1.copy()

    # --- Phase 4: tiny radius shrink to guarantee strict validity ---
    if best_arr is None:
        best_arr = _grid_initial()
    # shrink radii by a tiny epsilon to avoid boundary equality issues
    out = best_arr.copy()
    out[:, 2] *= (1.0 - 1e-9)
    # clamp centers to [r, 1-r]
    out[:, 0] = np.clip(out[:, 0], out[:, 2], 1.0 - out[:, 2])
    out[:, 1] = np.clip(out[:, 1], out[:, 2], 1.0 - out[:, 2])
    # re-verify; if invalid, fall back to grid
    if not _valid(out):
        out = _grid_initial()
        out[:, 2] *= (1.0 - 1e-9)
    return out
