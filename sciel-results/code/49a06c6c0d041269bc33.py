"""Packing 12 unit hexagons in a minimal regular hexagon: DE + Powell with time guard."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)
APO = SQRT3 / 2.0  # apothem of unit hexagon

# Unit flat-top hexagon: vertices (radius 1) and outward unit edge-normals
_HEX_V = np.array([
    [1.0, 0.0], [0.5, APO], [-0.5, APO],
    [-1.0, 0.0], [-0.5, -APO], [0.5, -APO],
])
_HEX_N = np.array([
    [SQRT3/2, 0.5], [0.0, 1.0], [-SQRT3/2, 0.5],
    [-SQRT3/2, -0.5], [0.0, -1.0], [SQRT3/2, -0.5],
])
_OUTER_N = _HEX_N  # outer hexagon (flat-top) unit normals

_DEADLINE = 115.0
_T0 = [0.0]

def _elapsed():
    return time.time() - _T0[0]

def _compute(x, y, a, lam):
    """Return (S, overlap_penalty). x,y,a are (N,) arrays."""
    ca = np.cos(a); sa = np.sin(a)
    vx = _HEX_V[:, 0]; vy = _HEX_V[:, 1]
    # all vertices (N,6,2)
    all_vx = ca[:, None] * vx[None, :] - sa[:, None] * vy[None, :] + x[:, None]
    all_vy = sa[:, None] * vx[None, :] + ca[:, None] * vy[None, :] + y[:, None]
    # S: outer center origin, outer rotation 0
    on0 = _OUTER_N[:, 0]; on1 = _OUTER_N[:, 1]
    dots = all_vx[:, :, None] * on0[None, None, :] + all_vy[:, :, None] * on1[None, None, :]
    S = float(np.max(dots)) / APO
    if S < 0.0:
        S = 0.0
    if lam == 0.0:
        return S, 0.0
    # normals (N,6,2)
    nx = _HEX_N[:, 0]; ny = _HEX_N[:, 1]
    all_nx = ca[:, None] * nx[None, :] - sa[:, None] * ny[None, :]
    all_ny = sa[:, None] * nx[None, :] + ca[:, None] * ny[None, :]
    pen = 0.0
    for i in range(N):
        vi_x = all_vx[i]; vi_y = all_vy[i]
        ni_x = all_nx[i]; ni_y = all_ny[i]
        for j in range(i + 1, N):
            ax = np.empty(12); ay = np.empty(12)
            ax[:6] = ni_x; ay[:6] = ni_y
            ax[6:] = all_nx[j]; ay[6:] = all_ny[j]
            pi = vi_x[:, None] * ax[None, :] + vi_y[:, None] * ay[None, :]  # (6,12)
            pj = all_vx[j, :, None] * ax[None, :] + all_vy[j, :, None] * ay[None, :]
            min_i = pi.min(axis=0); max_i = pi.max(axis=0)
            min_j = pj.min(axis=0); max_j = pj.max(axis=0)
            sep = np.maximum(min_i - max_j, min_j - max_i).max()
            if sep < 0.0:
                d = -sep
                pen += d * d
    return S, lam * pen

def _objective(v, lam):
    if _elapsed() > _DEADLINE:
        raise TimeoutError("budget")
    x = v[0:N]; y = v[N:2*N]; a = v[2*N:3*N]
    S, pen = _compute(x, y, a, lam)
    return S + pen

def _is_valid(x, y, a, tol=1e-6):
    S, raw_pen = _compute(x, y, a, 1.0)
    return raw_pen < tol, S

def _fallback():
    # guaranteed-valid: 2 rows x 6, wide spacing
    inner = []
    for r in range(2):
        for c in range(6):
            inner.append([3.2 * (c - 2.5), 2.9 * r, 0.0])
    return inner, (0.0, 0.0), 22.0, 0.0

def construct():
    _T0[0] = time.time()
    rng = np.random.default_rng(42)
    lo = np.concatenate([np.full(N, -9.0), np.full(N, -9.0), np.zeros(N)])
    hi = np.concatenate([np.full(N, 9.0), np.full(N, 9.0), np.full(N, np.pi / 3.0)])
    best = None
    best_f = np.inf

    # Phase 1: global search
    try:
        res = differential_evolution(
            lambda v: _objective(v, 5.0),
            bounds=list(zip(lo, hi)),
            seed=42, maxiter=120, popsize=14, tol=1e-5,
            polish=False, init='sobol',
        )
        if res.x is not None and res.fun < best_f:
            best_f = float(res.fun); best = np.asarray(res.x, dtype=float)
    except Exception:
        pass

    # Phase 2: local refinement (stronger penalty)
    cands = []
    if best is not None:
        cands.append(best.copy())
    for _ in range(4):
        cands.append(rng.uniform(lo, hi))
    for c0 in cands:
        if _elapsed() > _DEADLINE - 8:
            break
        try:
            r2 = minimize(lambda v: _objective(v, 30.0), c0, method='Powell',
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            if r2.x is not None and r2.fun < best_f:
                best_f = float(r2.fun); best = np.asarray(r2.x, dtype=float)
        except Exception:
            pass

    if best is None:
        return _fallback()

    x = best[0:N]; y = best[N:2*N]; a = best[2*N:3*N]
    ok, S = _is_valid(x, y, a)
    if not ok:
        # try one more high-penalty polish
        try:
            r3 = minimize(lambda v: _objective(v, 200.0), best, method='Powell',
                          options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10})
            if r3.x is not None:
                x = r3.x[0:N]; y = r3.x[N:2*N]; a = r3.x[2*N:3*N]
                ok, S = _is_valid(x, y, a)
        except Exception:
            pass
    if not ok:
        return _fallback()

    inner = [[float(x[i]), float(y[i]), float(np.degrees(a[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0
