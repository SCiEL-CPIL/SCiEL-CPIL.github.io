"""Packing 12 unit hexagons in a minimal regular hexagon via Differential Evolution."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
T0 = time.time()
DEADLINE = 150.0
MARGIN = 25.0

# Unit regular hexagon vertices (flat-top), centered at origin, side length 1.
# Vertices at angles 0,60,...,300 degrees on the circumcircle of radius 1.
_VERT = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)])
_COS = np.array([np.cos(np.deg2rad(a)) for a in range(0, 360, 60)])
_SIN = np.array([np.sin(np.deg2rad(a)) for a in range(0, 360, 60)])
# The 3 unique SAT normal directions for a hexagon (opposite edges share a normal).
_UX = np.array([np.cos(np.deg2rad(a)) for a in (90, 30, -30)])
_UY = np.array([np.sin(np.deg2rad(a)) for a in (90, 30, -30)])
# Support function of the unit hexagon in directions (ux,uy) and (-ux,-uy).
_SUPP = np.max(_VERT * _UX[None, :], axis=1)
_NEGSUPP = np.max(_VERT * (-_UX[None, :]), axis=1)


def _time_ok():
    return (time.time() - T0) < (DEADLINE - MARGIN)


def _sat_gap_sq(a1, b1, a2, b2):
    """Return squared overlap depth between two translated+rotated unit hexagons.
    a1,b1,a2,b2 are (x, y, theta) for the two hexagons. 0 means separated."""
    # Rotation matrices (rows = images of e_x, e_y).
    r1 = np.array([[np.cos(a1), -np.sin(a1)], [np.sin(a1), np.cos(a1)]])
    r2 = np.array([[np.cos(a2), -np.sin(a2)], [np.sin(a2), np.cos(a2)]])
    # Rotated unit-hexagon vertices.
    v1 = _VERT @ r1.T
    v2 = _VERT @ r2.T
    # SAT normals of each hexagon in world frame (3 unique each).
    n1 = _UX @ r1.T
    n2 = _UX @ r2.T
    # Support of each hexagon in each normal direction.
    s11 = np.max(v1 * n1[None, :], axis=1)
    s12 = np.max(v1 * n2[None, :], axis=1)
    s21 = np.max(v2 * n1[None, :], axis=1)
    s22 = np.max(v2 * n2[None, :], axis=1)
    d = a2 - a1
    d1 = d @ n1.T
    d2 = d @ n2.T
    # Required gap on each normal: (support of hex1 in +n) + (support of hex2 in -n).
    req1 = s11 + _NEGSUPP
    req2 = s12 + _NEGSUPP
    # Signed overlap depth (positive => overlap) on each normal.
    o1 = req1 - d1
    o2 = req2 - d2
    depth = max(float(np.max(o1)), float(np.max(o2)), 0.0)
    return depth * depth


def _objective(flat, penalty, outer_theta):
    """Objective: outer hexagon side length + penalty * (total overlap depth^2)."""
    pos = flat.reshape(N, 3)
    P = pos[:, 0]
    Q = pos[:, 1]
    T = pos[:, 2]
    # Outer hexagon support in its 3 normal directions.
    ct, st = np.cos(outer_theta), np.sin(outer_theta)
    R = np.array([[ct, -st], [st, ct]])
    n = _UX @ R.T
    nx = n[:, 0]
    ny = n[:, 1]
    # Side length = max over vertices of (support in +n) + (support in -n).
    sup = np.empty((3, N))
    for i in range(3):
        sup[i] = np.abs(P * nx[i] + Q * ny[i])
    S = float(np.max(sup.sum(axis=1)))
    # Pairwise overlap penalty (SAT).
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            pen += _sat_gap_sq(T[i], (P[i], Q[i]), T[j], (P[j], Q[j]))
    return S + penalty * pen


def _local_polish(x0, outer_theta):
    """Refine the best DE solution with a smooth local optimizer."""
    def obj(flat):
        return _objective(flat, 1e6, outer_theta)
    best_x = np.array(x0, dtype=float)
    best_f = obj(best_x)
    for _ in range(3):
        if not _time_ok():
            break
        try:
            res = minimize(obj, best_x, method="Nelder-Mead",
                           options={"maxiter": 800, "xatol": 1e-9, "fatol": 1e-11})
        except Exception:
            break
        if res.fun < best_f:
            best_f = float(res.fun)
            best_x = res.x
    return best_x, best_f


def construct():
    # Fixed outer orientation (hexagon has 6-fold symmetry; 0 is without loss of generality).
    outer_theta = 0.0
    # Initial side-length estimate: 12 unit hexagons in a row need ~ 12*2 = 24 width,
    # but a 2D cluster is far tighter. Use a generous bound for DE bounds.
    bound = 14.0

    # Build DE bounds: x, y in [-bound, bound], angle in [0, 2*pi).
    lb = []
    ub = []
    for _ in range(N):
        lb += [-bound, -bound, 0.0]
        ub += [bound, bound, 2.0 * np.pi]
    lb = np.array(lb)
    ub = np.array(ub)

    # Random seed for reproducibility of the search (not required, but stable).
    rng = np.random.default_rng(12345)

    # Phase 1: Differential Evolution global search (robust to the non-smooth penalty).
    best_x = None
    best_f = np.inf
    # Adaptive penalty schedule: start moderate, increase to enforce non-overlap.
    for penalty in (1e3, 1e4, 1e5):
        if not _time_ok():
            break
        try:
            res = differential_evolution(
                _objective, bounds=list(zip(lb, ub)), args=(penalty, outer_theta),
                seed=int(rng.integers(0, 2**31)), maxiter=60, popsize=12,
                tol=1e-8, mutation=(0.3, 0.9), recombination=0.7,
                polish=False, init="latinhypercube",
                callback=lambda xk, convergence: (not _time_ok()) and 1 or 0,
            )
        except Exception:
            continue
        f = float(res.fun)
        if f < best_f:
            best_f = f
            best_x = res.x

    if best_x is None:
        # Fallback: a valid (if not optimal) linear-chain packing.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Phase 2: Local polish to squeeze precision and remove residual overlap.
    if _time_ok():
        best_x, _ = _local_polish(best_x, outer_theta)

    pos = best_x.reshape(N, 3)
    # Final outer side length (no penalty term).
    ct, st = np.cos(outer_theta), np.sin(outer_theta)
    R = np.array([[ct, -st], [st, ct]])
    n = _UX @ R.T
    P = pos[:, 0]
    Q = pos[:, 1]
    sup = np.empty((3, N))
    for i in range(3):
        sup[i] = np.abs(P * n[i, 0] + Q * n[i, 1])
    S = float(np.max(sup.sum(axis=1)))

    # Build the return interface: inner rows (x, y, angle_degrees).
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(np.rad2deg(pos[i, 2]))] for i in range(N)]
    # Center the whole configuration at the origin for cleanliness (does not change S).
    cx = float(np.mean(pos[:, 0]))
    cy = float(np.mean(pos[:, 1]))
    for row in inner:
        row[0] -= cx
        row[1] -= cy
    return inner, (cx, cy), S, float(np.rad2deg(outer_theta))
