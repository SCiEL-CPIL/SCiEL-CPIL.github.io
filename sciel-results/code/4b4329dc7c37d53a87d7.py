# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_T0 = time.time()
_DEADLINE = 560.0  # safety margin before 600s


def _hex_vertices(cx, cy, s, a):
    """Vertices of a regular hexagon (center cx,cy, side s, angle a in radians)."""
    th = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _separation_ok(hexes):
    """Return True if every pair of unit hexagons is strictly separated (>1e-6 gap
    on at least one of the 12 edge normals)."""
    n = len(hexes)
    for i in range(n):
        vi = _hex_vertices(hexes[i, 0], hexes[i, 1], 1.0, np.deg2rad(hexes[i, 2]))
        for j in range(i + 1, n):
            vj = _hex_vertices(hexes[j, 0], hexes[j, 1], 1.0, np.deg2rad(hexes[j, 2]))
            ok = False
            for k in range(6):
                p1, p2 = vi[k], vi[(k + 1) % 6]
                edge = p2 - p1
                nrm = np.array([-edge[1], edge[0]])
                nrm /= (np.linalg.norm(nrm) + 1e-15)
                a1 = np.dot(vi, nrm)
                a2 = np.dot(vj, nrm)
                if (a1.max() < a2.min() - 1e-6) or (a2.max() < a1.min() - 1e-6):
                    ok = True
                    break
            if not ok:
                return False
    return True


def _outer_side(hexes):
    """Minimal side length of a regular hexagon (center 0, angle 0) that contains
    all inner hexagons (all inner vertices satisfy the cross-product containment)."""
    vs = []
    for h in hexes:
        vs.append(_hex_vertices(h[0], h[1], 1.0, np.deg2rad(h[2])))
    vs = np.vstack(vs)
    ev = _hex_vertices(0.0, 0.0, 1.0, 0.0)
    lo = 0.0
    hi = 200.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        oev = _hex_vertices(0.0, 0.0, mid, 0.0)
        ok = True
        for k in range(6):
            p1, p2 = oev[k], oev[(k + 1) % 6]
            e = p2 - p1
            c = np.cross(e, vs - p1)
            if np.any(c < -1e-6):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def _objective(x):
    """Objective = outer side length (minimized). Invalid configs -> large penalty."""
    hexes = x.reshape(11, 3)
    if not np.all(np.isfinite(hexes)):
        return 1e6
    if not _separation_ok(hexes):
        return 1e6
    return _outer_side(hexes)


def _nm_refine(x):
    try:
        res = minimize(_objective, x, method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-10})
        return res.x
    except Exception:
        return x


def _build_result(hexes):
    hexes = np.asarray(hexes, dtype=float).reshape(11, 3)
    side = _outer_side(hexes)
    inner = hexes.copy()
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


def _fallback():
    inner = np.array([
        [0.0, 0.0, 0.0],
        [-2.5, 0.0, 0.0], [2.5, 0.0, 0.0],
        [-1.25, 2.17, 0.0], [1.25, 2.17, 0.0],
        [-1.25, -2.17, 0.0], [1.25, -2.17, 0.0],
        [-3.75, 2.17, 0.0], [3.75, 2.17, 0.0],
        [-3.75, -2.17, 0.0], [3.75, -2.17, 0.0],
    ])
    return _build_result(inner)


def hexagon_packing_11():
    # Symmetry-aware starting layout (center + 6 around + 4 more).
    a = np.sqrt(3.0)
    base = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.7320508, 0.0], [1.5, 0.8660254, 0.0], [1.5, -0.8660254, 0.0],
        [0.0, -1.7320508, 0.0], [-1.5, -0.8660254, 0.0], [-1.5, 0.8660254, 0.0],
        [0.0, 3.4641016, 0.0], [3.0, 1.7320508, 0.0],
        [-3.0, 1.7320508, 0.0], [3.0, -1.7320508, 0.0],
    ])
    best_x = base.ravel().copy()
    best_f = _objective(best_x)

    bounds = [(-6.0, 6.0), (-6.0, 6.0), (-180.0, 180.0)] * 11

    # Global search: differential evolution.
    try:
        de_res = differential_evolution(
            _objective, bounds, seed=1, maxiter=120, popsize=18,
            tol=1e-9, mutation=(0.4, 1.0), recombination=0.8,
            polish=False, init='sobol', updating='deferred',
            callback=lambda xk, con: (_ for _ in ()).throw(
                Exception('stop')) if (time.time() - _T0) > _DEADLINE else None,
        )
        if de_res.x is not None and np.all(np.isfinite(de_res.x)):
            if de_res.fun < best_f:
                best_f = de_res.fun
                best_x = de_res.x
    except Exception:
        pass

    # Local refinement: Nelder-Mead on the best candidate.
    if (time.time() - _T0) < _DEADLINE:
        cand = _nm_refine(best_x)
        cf = _objective(cand)
        if cf < best_f:
            best_f = cf
            best_x = cand

    # A couple of independent restarts if time remains.
    for restart in range(3):
        if (time.time() - _T0) > _DEADLINE - 60:
            break
        try:
            r = differential_evolution(
                _objective, bounds, seed=100 + restart, maxiter=60, popsize=14,
                tol=1e-9, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='sobol', updating='deferred',
            )
            if r.x is not None and np.all(np.isfinite(r.x)):
                if r.fun < best_f:
                    best_f = r.fun
                    best_x = r.x
                if (time.time() - _T0) < _DEADLINE - 20:
                    cand = _nm_refine(r.x)
                    cf = _objective(cand)
                    if cf < best_f:
                        best_f = cf
                        best_x = cand
        except Exception:
            pass

    return _build_result(best_x)


# EVOLVE-BLOCK-END
