"""Packing 12 unit regular hexagons in a regular hexagon (minimize outer side)."""
import time
import numpy as np

_N = 12
_R = 1.0            # circumradius of a unit hexagon (side = 1)
_TAU = 2.0 * np.pi
_EPS = 1e-9

# 6 edge normals of a unit hexagon, starting at 30 deg, every 60 deg
_NORM = np.array([[np.cos(np.deg2rad(a)) for a in (30, 90, 150, 210, 270, 330)]],
                 dtype=float)          # (1,6)
_COSN = np.cos(_NORM)                  # (1,6)
_SINN = np.sin(_NORM)                  # (1,6)

# Precompute all 6*6 axis pairs
_AX = np.array([np.cos(np.deg2rad(a)) for a in (30, 90, 150, 210, 270, 330)])
_AY = np.array([np.sin(np.deg2rad(a)) for a in (30, 90, 150, 210, 270, 330)])
_AX2 = _AX[:, None] * _AX[None, :]      # (6,6)
_AY2 = _AY[:, None] * _AY[None, :]
_AXAY = _AX[:, None] * _AY[None, :]     # (6,6)
_AYAX = _AY[:, None] * _AX[None, :]
_COS2 = np.cos(_NORM)[:, None] * np.cos(_NORM)[None, :]   # (6,6)
_SIN2 = np.sin(_NORM)[:, None] * np.sin(_NORM)[None, :]   # (6,6)
_COS2T = np.cos(_NORM)[None, :] * np.cos(_NORM)[:, None]  # (6,6)
_SIN2T = np.sin(_NORM)[None, :] * np.sin(_NORM)[:, None]  # (6,6)


def _vertices(px, py, a):
    """(N,2) array of the 6 vertices of a unit hexagon centered at (px,py), rotated a."""
    ang = a[None, :] + np.deg2rad([0, 60, 120, 180, 240, 300])
    c = np.cos(ang)
    s = np.sin(ang)
    return np.stack([px + c, py + s], axis=-1)


def _sat_sep(p1, p2, a1, a2):
    """Minimal translation distance so the two unit hexagons just touch (>=0)."""
    v1 = _vertices(p1, p2, a1)
    v2 = _vertices(p1, p2, a2)
    c1 = np.cos(a1[None, :])
    s1 = np.sin(a1[None, :])
    c2 = np.cos(a2[None, :])
    s2 = np.sin(a2[None, :])
    c12 = c1 * c2 + s1 * s2
    s12 = s1 * c2 - c1 * s2
    c21 = c12
    s21 = -s12
    # (N,6,6) projections
    p11 = v1[:, :, None, 0] * _COS2 + v1[:, :, None, 1] * _SIN2
    p12 = v1[:, None, :, 0] * _COS2T + v1[:, None, :, 1] * _SIN2T
    p21 = v2[:, :, None, 0] * _COS2 + v2[:, :, None, 1] * _SIN2
    p22 = v2[:, None, :, 0] * _COS2T + v2[:, None, :, 1] * _SIN2T
    d1 = p11.max(-1) - p22.min(-1)   # (N,6)
    d2 = p12.min(-1) - p21.max(-1)   # (N,6)
    return np.maximum(0.0, np.maximum(d1, d2).max(-1))


def _boundary(p, S):
    """Max over hexes of (max vertex distance from origin - S)."""
    d = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    return (d + _R - S).max()


def _objective(z, lam=100.0, bnd=1000.0):
    n = _N
    p = z[0:2 * n].reshape(n, 2)
    a = z[2 * n:3 * n]
    S = z[3 * n]
    sep = np.zeros(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            sep[k] = _sat_sep(p[i, 0], p[i, 1], a[i], a[j])
            k += 1
    viol = (sep < 0.0).sum()
    if viol == 0:
        return S + bnd * _boundary(p, S)
    return S + lam * viol + bnd * _boundary(p, S)


def _refine(z, iters=80, step=0.02):
    """Local coordinate-descent refinement of a feasible configuration (reduces S)."""
    z = np.array(z, dtype=float)
    n = _N
    best = _objective(z)
    for _ in range(iters):
        improved = False
        for i in range(3 * n + 1):
            for ds in (step, -step):
                z2 = z.copy()
                z2[i] += ds
                v = _objective(z2)
                if v < best - 1e-12:
                    z = z2
                    best = v
                    improved = True
        if not improved:
            break
    return z, best


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n = _N
    rng = np.random.default_rng(20240517)

    def _z_from(pos, ang, S):
        z = np.zeros(3 * n + 1)
        z[0:2 * n] = pos.reshape(-1)
        z[2 * n:3 * n] = ang
        z[3 * n] = S
        return z

    def _best_z0():
        cands = []
        # hexagonal lattice
        gx = np.sqrt(3.0)
        gy = 1.5
        pts = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                pts.append((gx * q + 0.5 * gy * r, gy * r))
        pts = np.array(pts)
        d = np.hypot(pts[:, 0], pts[:, 1])
        order = np.argsort(d)
        for cnt in (7, 13, 12):
            sel = order[:cnt]
            cands.append(_z_from(pts[sel], np.zeros(cnt), d[sel].max() + _R + 0.05))
        # ring + center
        ring = np.stack([np.cos(np.linspace(0, _TAU, 12, endpoint=False)),
                         np.sin(np.linspace(0, _TAU, 12, endpoint=False))], axis=-1) * 2.1
        pos = np.vstack([np.zeros(2), ring])
        cands.append(_z_from(pos, np.zeros(n), 2.1 + _R + 0.05))
        # two rings
        r1 = np.stack([np.cos(np.linspace(0, _TAU, 6, endpoint=False)),
                       np.sin(np.linspace(0, _TAU, 6, endpoint=False))], axis=-1) * 2.0
        r2 = np.stack([np.cos(np.linspace(0, _TAU, 6, endpoint=False) + np.pi / 6),
                       np.sin(np.linspace(0, _TAU, 6, endpoint=False) + np.pi / 6)], axis=-1) * 4.0
        pos = np.vstack([r1, r2])
        cands.append(_z_from(pos, np.zeros(n), 4.0 + _R + 0.05))
        # random
        for _ in range(8):
            pos = rng.normal(0, 2.5, (n, 2))
            ang = rng.uniform(0, np.pi, n)
            S = np.hypot(pos[:, 0], pos[:, 1]).max() + _R + 0.5
            cands.append(_z_from(pos, ang, S))
        best = min(cands, key=lambda z: _objective(z))
        return best

    best_z = _best_z0()
    best_val = _objective(best_z)

    def _accept(z, val):
        nonlocal best_z, best_val
        if val < best_val:
            best_z, best_val = z, val

    # Phase 1: global L-BFGS-B with several restarts
    while time.time() < deadline - 25.0:
        z0 = best_z + rng.normal(0, 0.35, 3 * n + 1)
        z0[3 * n] = best_z[3 * n] + rng.uniform(-0.5, 1.0)
        try:
            res = None
            try:
                from scipy.optimize import minimize
                res = minimize(_objective, z0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-9})
            except Exception:
                res = None
            if res is not None and hasattr(res, 'x'):
                _accept(res.x, _objective(res.x))
        except Exception:
            pass

    # Phase 2: local refinement of the best configuration
    while time.time() < deadline - 5.0:
        try:
            zr, vr = _refine(best_z, iters=60, step=0.02)
            _accept(zr, vr)
        except Exception:
            break

    z = best_z
    pos = z[0:2 * n].reshape(n, 2)
    ang = z[2 * n:3 * n]
    S = float(z[3 * n])
    S = max(S, float(np.hypot(pos[:, 0], pos[:, 1]).max() + _R))  # ensure boundary feasibility

    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(np.degrees(ang[i]))] for i in range(n)]
    return inner, (0.0, 0.0), float(S), 0.0
