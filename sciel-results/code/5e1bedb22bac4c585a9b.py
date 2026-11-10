import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 600.0
_MARGIN = 0.02          # safety margin for separation / containment
_SEP_TOL = 1e-6
_CONT_TOL = 1e-6
_H = np.sqrt(3.0)

# 6 edge normals (unit) for a flat-top unit hexagon, i=0..5
_NRM = np.array([
    [1.0, 0.0],
    [0.5, 0.5 * _H],
    [-0.5, 0.5 * _H],
    [-1.0, 0.0],
    [-0.5, -0.5 * _H],
    [0.5, -0.5 * _H],
])
_NRM = _NRM / np.linalg.norm(_NRM, axis=1, keepdims=True)
# vertex offsets for a unit flat-top hexagon (i=0..5)
_VTX = np.array([
    [1.0, 0.0],
    [0.5, 0.5 * _H],
    [-0.5, 0.5 * _H],
    [-1.0, 0.0],
    [-0.5, -0.5 * _H],
    [0.5, -0.5 * _H],
])


def _sym_centers(a, b, c):
    """Expand 3 independent centers to the full 6-fold symmetric set of 12."""
    c1 = np.array([a, 0.0])
    c2 = np.array([b, _H])
    c3 = np.array([c, 2.0 * _H])
    base = [c1, c2, c3]
    centers = []
    for k in range(6):
        th = np.deg2rad(60.0 * k)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        for p in base:
            centers.append(R @ p)
    return np.array(centers)


def _outer_side(centers):
    """Minimum enclosing regular hexagon (flat-top, centered at origin) side length."""
    return float(np.max(np.abs(centers[:, 0]) / 2.0 + np.abs(centers[:, 1]) / _H))


def _pair_sep(centers):
    """Minimum separation margin over all unordered pairs (SAT edge normals)."""
    n = centers.shape[0]
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[j] - centers[i]
            proj = _NRM @ d
            m = np.min(proj) + np.max(-proj)
            if m < best:
                best = m
    return best


def _cont_margin(centers, s):
    """Minimum containment margin (<=0 means some vertex outside the outer hexagon)."""
    best = np.inf
    for c in centers:
        for v in _VTX:
            p = c + v
            val = s - (np.abs(p[0]) / 2.0 + np.abs(p[1]) / _H)
            if val < best:
                best = val
    return best


def _objective(x):
    a, b, c = x
    centers = _sym_centers(a, b, c)
    return _outer_side(centers)


def _objective_pen(x):
    a, b, c = x
    centers = _sym_centers(a, b, c)
    s = _outer_side(centers)
    sep = _pair_sep(centers)
    cont = _cont_margin(centers, s)
    pen = 0.0
    if sep < _MARGIN:
        pen += 1e4 * (_MARGIN - sep)
    if cont < _MARGIN:
        pen += 1e4 * (_MARGIN - cont)
    return s + pen


def _is_valid(centers, s):
    if not np.all(np.isfinite(centers)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12.0) - 1e-9:
        return False
    if _pair_sep(centers) < _SEP_TOL:
        return False
    if _cont_margin(centers, s) < _CONT_TOL:
        return False
    return True


def _make_result(x, s):
    centers = _sym_centers(*x)
    return centers, np.array([0.0, 0.0, 0.0]), s


def hexagon_packing_12():
    global _T0
    _T0 = time.time()

    def elapsed():
        return time.time() - _T0

    # ---- seed points (a, b, c) ----
    seeds = [
        [2.0, 1.0, 1.0],
        [2.0, 0.9, 1.1],
        [2.0, 1.1, 0.9],
        [2.0, 1.0, 1.5],
        [2.0, 1.5, 1.0],
        [2.0, 0.8, 1.2],
        [2.0, 1.2, 0.8],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [2.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
    rng = np.random.default_rng(12345)
    for _ in range(24):
        seeds.append([
            rng.uniform(1.5, 2.5),
            rng.uniform(0.5, 1.6),
            rng.uniform(0.5, 1.6),
        ])

    bounds = [(1.0, 4.0), (0.2, 3.0), (0.2, 3.0)]

    best_x = None
    best_s = np.inf
    best_pen = np.inf

    def consider(x, use_penalty=False):
        nonlocal best_x, best_s, best_pen
        if not np.all(np.isfinite(x)):
            return
        centers = _sym_centers(*x)
        s = _outer_side(centers)
        if use_penalty:
            p = _objective_pen(x)
            if p < best_pen - 1e-12:
                best_pen = p
                best_x = x.copy()
                best_s = s
        else:
            if _is_valid(centers, s):
                if s < best_s - 1e-12:
                    best_s = s
                    best_x = x.copy()

    # ---- pass 1: L-BFGS-B with inequality constraints ----
    for seed in seeds:
        if elapsed() > _DEADLINE - 120.0:
            break
        x0 = np.array(seed, dtype=float)
        try:
            res = minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": lambda x: _pair_sep(_sym_centers(*x)) - _MARGIN},
                    {"type": "ineq", "fun": lambda x: _cont_margin(_sym_centers(*x), _outer_side(_sym_centers(*x))) - _MARGIN},
                ],
                options={"maxiter": 200, "maxfun": 4000, "ftol": 1e-12},
            )
            consider(res.x)
        except Exception:
            pass

    # ---- pass 2: Nelder-Mead on the penalty objective (more robust) ----
    for seed in seeds:
        if elapsed() > _DEADLINE - 60.0:
            break
        x0 = np.array(seed, dtype=float)
        try:
            res = minimize(
                _objective_pen,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 800, "xatol": 1e-9, "fatol": 1e-12, "adaptive": True},
            )
            consider(res.x, use_penalty=True)
            # also record if this penalty-optimal point happens to be valid
            consider(res.x)
        except Exception:
            pass

    # ---- pass 3: polish the current best with L-BFGS-B if valid ----
    if best_x is not None:
        if elapsed() < _DEADLINE - 30.0:
            try:
                res = minimize(
                    _objective,
                    best_x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    constraints=[
                        {"type": "ineq", "fun": lambda x: _pair_sep(_sym_centers(*x)) - _MARGIN},
                        {"type": "ineq", "fun": lambda x: _cont_margin(_sym_centers(*x), _outer_side(_sym_centers(*x))) - _MARGIN},
                    ],
                    options={"maxiter": 300, "maxfun": 6000, "ftol": 1e-13},
                )
                consider(res.x)
            except Exception:
                pass

    if best_x is None:
        # fallback: a known-good grid-like packing (guaranteed valid)
        centers = np.array([
            [0, 0],
            [-2.5, 0], [2.5, 0],
            [-1.25, _H], [1.25, _H],
            [-1.25, -_H], [1.25, -_H],
            [-3.75, _H], [3.75, _H],
            [-3.75, -_H], [3.75, -_H],
            [0, -2.0 * _H],
        ])
        s = _outer_side(centers)
        return centers, np.array([0.0, 0.0, 0.0]), s

    centers, outer, s = _make_result(best_x, best_s)
    return centers, outer, s
