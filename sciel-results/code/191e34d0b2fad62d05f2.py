"""Rewritten program: centrally-symmetric Heilbronn n=14 optimization.

Exploits the known structural property that optimal configurations for even N
are centrally symmetric (P[i] = -P[i+7]), reducing the search from 28 to 14
DOF. Multi-start (Nelder-Mead + Powell + L-BFGS-B on soft-min objective) over
diverse initializations, with a final high-precision polish.
"""
N = 14
M = N // 2  # 7

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

_T0 = time.time()
_DEADLINE = _T0 + 142.0  # safety margin below 150s kill

# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area (not doubled) over all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j + 1:] - x[i]
            dy_ki = y[j + 1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = float(np.min(areas))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _hull_area_fast(points):
    """Convex hull area; 0 if degenerate."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """min triangle area / hull area (the reported metric)."""
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-14:
        return 0.0
    return min_area / hull_area


def _all_triangle_areas(points):
    """Flat array of all triangle areas (not doubled)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j + 1:] - x[i]
            dy_ki = y[j + 1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            tri = 0.5 * np.abs(cross)
            if tri.size:
                parts.append(tri)
    if not parts:
        return np.array([0.0])
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Symmetric parameterization: optimize 7 points, mirror to get 14
# ---------------------------------------------------------------------------

def _full_from_half(half):
    """Build the 14-point centrally-symmetric set from 7 points."""
    half = np.asarray(half, dtype=float).reshape(M, 2)
    return np.vstack([half, -half])


def _objective_hard_sym(half_flat):
    """Minimize hull_area / min_triangle_area for the symmetric set."""
    if np.any(np.isnan(half_flat)) or np.any(np.isinf(half_flat)):
        return 1e10
    pts = _full_from_half(half_flat)
    min_area = _min_triangle_area_vec(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft_sym(half_flat, beta=150.0):
    """Differentiable soft-min surrogate: hull_area / soft_min(areas)."""
    if np.any(np.isnan(half_flat)) or np.any(np.isinf(half_flat)):
        return 1e10
    pts = _full_from_half(half_flat)
    areas = _all_triangle_areas(pts)
    if areas.size == 0 or float(np.min(areas)) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = float(np.max(z))
    if mz < -500:
        log_sum = mz
    else:
        log_sum = mz + float(np.log(np.sum(np.exp(z - mz))))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _objective_soft_medium_sym(half_flat, beta=60.0):
    """Broader-exploration soft-min surrogate."""
    if np.any(np.isnan(half_flat)) or np.any(np.isinf(half_flat)):
        return 1e10
    pts = _full_from_half(half_flat)
    areas = _all_triangle_areas(pts)
    if areas.size == 0 or float(np.min(areas)) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = float(np.max(z))
    if mz < -500:
        log_sum = mz
    else:
        log_sum = mz + float(np.log(np.sum(np.exp(z - mz))))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_unit_area(points):
    """Scale so convex hull area = 1."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        return points / np.sqrt(w * h)
    return points


# ---------------------------------------------------------------------------
# Optimizer wrappers (with deadline guards)
# ---------------------------------------------------------------------------

def _nm(x0, deadline, max_iter=3000, obj_func=None):
    if obj_func is None:
        obj_func = _objective_hard_sym
    if time.time() > deadline - 2.0:
        return None
    try:
        res = minimize(obj_func, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10,
                                'fatol': 1e-12, 'adaptive': True})
        return res.x
    except Exception:
        return None


def _powell(x0, deadline, max_iter=1000, obj_func=None):
    if obj_func is None:
        obj_func = _objective_hard_sym
    if time.time() > deadline - 3.0:
        return None
    try:
        res = minimize(obj_func, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-12,
                                'ftol': 1e-14})
        return res.x
    except Exception:
        return None


def _lbfgs(x0, deadline, max_iter=600, obj_func=None):
    if obj_func is None:
        obj_func = _objective_soft_sym
    if time.time() > deadline - 3.0:
        return None
    try:
        res = minimize(obj_func, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14,
                                'gtol': 1e-11, 'maxfun': 60000})
        return res.x
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Initializations (all in the 7-point half-space, mirrored to 14)
# ---------------------------------------------------------------------------

def _make_inits():
    inits = []
    rng = np.random.default_rng(42)

    # Regular 14-gon -> take 7 consecutive points as the half
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    full = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(full[:M].flatten())

    # Rotated regular 14-gon
    for rot in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        a = angles + rot
        f = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(f[:M].flatten())

    # Jittered regular
    for js in [0.05, 0.1, 0.15, 0.25, 0.4]:
        jit = rng.normal(0, js, size=(M, 2))
        inits.append((full[:M] + jit).flatten())

    # Elliptical 14-gon (stretch x)
    for ax in [1.2, 1.4, 1.6, 1.9, 2.2]:
        f = np.column_stack([ax * np.cos(angles), np.sin(angles)])
        inits.append(f[:M].flatten())

    # Two concentric rings (7 inner + 7 outer) -> take 7 of them
    for r_in, n_in in [(0.4, 4), (0.5, 5), (0.6, 6), (0.45, 3)]:
        n_out = M - n_in
        a_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + np.pi / max(n_out, 1)
        p_in = r_in * np.column_stack([np.cos(a_in), np.sin(a_in)])
        p_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
        pts = np.vstack([p_in, p_out])
        inits.append(pts.flatten())

    # Random convex-like: random radii on sorted angles
    for seed in range(10):
        rs = np.random.default_rng(seed * 101 + 3)
        ang = np.sort(rs.uniform(0, 2 * np.pi, M))
        rad = rs.uniform(0.6, 1.3, M)
        pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        inits.append(pts.flatten())

    # Fibonacci spiral (7 points)
    ga = 2.399963229728653
    for rs in [1.0, 1.3]:
        pts = np.zeros((M, 2))
        for i in range(M):
            ang = i * ga
            r = rs * np.sqrt((i + 0.5) / M)
            pts[i, 0] = r * np.cos(ang)
            pts[i, 1] = r * np.sin(ang)
        inits.append(pts.flatten())

    # Random points clipped to unit disk, sorted by angle
    for seed in range(10):
        rs = np.random.default_rng(seed * 555 + 7)
        raw = rs.uniform(-1, 1, size=(M, 2))
        norms = np.linalg.norm(raw, axis=1)
        mask = norms > 1
        raw[mask] = raw[mask] / norms[mask, None]
        ang = np.arctan2(raw[:, 1], raw[:, 0])
        raw = raw[np.argsort(ang)]
        inits.append(raw.flatten())

    # Alternating radii on regular 14-gon (7 points)
    for alt in [0.1, 0.2, 0.3]:
        a7 = np.linspace(0, 2 * np.pi, M, endpoint=False)
        rad = np.ones(M)
        rad[::2] = 1.0 + alt
        rad[1::2] = 1.0 - alt
        pts = np.column_stack([rad * np.cos(a7), rad * np.sin(a7)])
        inits.append(pts.flatten())

    return inits


# ---------------------------------------------------------------------------
# Final high-precision polish
# ---------------------------------------------------------------------------

def _final_polish(half_flat, deadline):
    best_obj = _objective_hard_sym(half_flat)
    best = half_flat.copy()

    def _try(x, label):
        nonlocal best_obj, best
        if x is None:
            return
        o = _objective_hard_sym(x)
        if o < best_obj:
            best_obj = o
            best = x.copy()

    # L-BFGS-B on soft objective, very tight
    if time.time() < deadline - 12:
        try:
            r = minimize(_objective_soft_sym, best, method='L-BFGS-B',
                         options={'maxiter': 3000, 'ftol': 1e-16,
                                  'gtol': 1e-13, 'maxfun': 200000})
            _try(r.x, 'lbfgs')
        except Exception:
            pass

    # Powell on hard objective, tight
    if time.time() < deadline - 10:
        r = _powell(best, deadline, max_iter=6000)
        _try(r, 'powell')

    # Nelder-Mead on hard, tight
    if time.time() < deadline - 8:
        r = _nm(best, deadline, max_iter=6000)
        _try(r, 'nm')

    # L-BFGS-B on medium-temperature soft
    if time.time() < deadline - 8:
        try:
            r = minimize(_objective_soft_medium_sym, best, method='L-BFGS-B',
                         options={'maxiter': 1500, 'ftol': 1e-15,
                                  'gtol': 1e-12, 'maxfun': 80000})
            _try(r.x, 'lbfgs_med')
        except Exception:
            pass

    # Final Powell
    if time.time() < deadline - 6:
        r = _powell(best, deadline, max_iter=3000)
        _try(r, 'powell2')

    return best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    start = time.time()
    deadline = start + 140.0

    inits = _make_inits()

    best_x = None
    best_obj = float('inf')

    def _consider(x):
        nonlocal best_x, best_obj
        if x is None:
            return
        o = _objective_hard_sym(x)
        if o < best_obj:
            best_obj = o
            best_x = x.copy()

    # Phase 1: multi-start Nelder-Mead (hard) + soft-L-BFGS per init
    n_init = len(inits)
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time()
        budget = min(4.0, max(1.5, remaining / max(1, n_init - idx)))
        r = _nm(x0, deadline, max_iter=int(budget * 900))
        _consider(r)
        if time.time() < deadline - 18:
            r2 = _lbfgs(x0, deadline - 18, max_iter=400, obj_func=_objective_soft_sym)
            _consider(r2)

    # Phase 2: differential evolution for global exploration (14 DOF is cheap)
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0)] * (2 * M)
            de_res = differential_evolution(
                _objective_soft_medium_sym, bounds,
                maxiter=40, popsize=12, seed=1234, tol=1e-5,
                init='sobol', polish=False,
                updating='immediate',
                maxfees=None,
            )
            _consider(de_res.x)
        except Exception:
            pass

    if best_x is None:
        best_x = inits[0].copy()

    # Phase 3: Powell polish
    if time.time() < deadline - 25:
        r = _powell(best_x, deadline, max_iter=3000)
        _consider(r)

    # Phase 4: L-BFGS-B soft
    if time.time() < deadline - 22:
        r = _lbfgs(best_x, deadline, max_iter=2000, obj_func=_objective_soft_sym)
        _consider(r)
        if time.time() < deadline - 20:
            try:
                r = minimize(_objective_soft_medium_sym, best_x, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-14,
                                      'gtol': 1e-11, 'maxfun': 60000})
                _consider(r.x)
            except Exception:
                pass

    # Phase 5: Nelder-Mead
    if time.time() < deadline - 18:
        r = _nm(best_x, deadline, max_iter=4000)
        _consider(r)

    # Phase 6: Powell again
    if time.time() < deadline - 15:
        r = _powell(best_x, deadline, max_iter=2000)
        _consider(r)

    # Phase 7: high-precision final polish
    if time.time() < deadline - 12:
        best_x = _final_polish(best_x, deadline)
        best_obj = _objective_hard_sym(best_x)

    # Build full 14-point set and normalize to unit hull area
    final_points = _full_from_half(best_x)
    final_points = _normalize_unit_area(final_points)

    # Safety checks
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            reg = np.column_stack([np.cos(angles), np.sin(angles)])
            final_points = _normalize_unit_area(reg)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg = np.column_stack([np.cos(angles), np.sin(angles)])
        final_points = _normalize_unit_area(reg)

    if np.any(np.isnan(final_points)) or np.any(np.isinf(final_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg = np.column_stack([np.cos(angles), np.sin(angles)])
        final_points = _normalize_unit_area(reg)

    return final_points
# EVOLVE-BLOCK-END
