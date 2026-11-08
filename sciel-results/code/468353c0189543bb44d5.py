"""Hexagon packing: 11 unit hexagons in a regular hexagon.
Joint continuous optimization via Nelder-Mead with lattice multi-starts.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 138.0
SQRT3 = np.sqrt(3.0)
HALF_SQRT3 = SQRT3 / 2.0
N_INNER = 6 * N  # 66


def _hex_verts(c, ang_rad, side=1.0):
    a = ang_rad + np.arange(6) * (np.pi / 3.0)
    return c[None, :] + side * np.column_stack([np.cos(a), np.sin(a)])


def _outer_normals(outer_rad):
    a = outer_rad + np.pi / 6.0 + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([np.cos(a), np.sin(a)])


def _inner_normals(ang_rad):
    a = ang_rad + np.pi / 6.0 + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([np.cos(a), np.sin(a)])


def _outer_side_for(centers, inner_rads, outer_rad):
    """Return (side, center) for the min enclosing outer hexagon."""
    allv = np.empty((N_INNER, 2))
    for i in range(N):
        allv[6*i:6*i+6] = _hex_verts(centers[i], inner_rads[i])
    on = _outer_normals(outer_rad)
    P = allv @ on.T
    mx = P.max(axis=0)
    mn = P.min(axis=0)
    widths = mx - mn
    c = 0.5 * (mx + mn) @ on
    return (widths.max() / SQRT3), c


def _outer_side_fixed_c(centers, inner_rads, outer_rad, c):
    """Side length with fixed center c (cheaper, no center solve)."""
    allv = np.empty((N_INNER, 2))
    for i in range(N):
        allv[6*i:6*i+6] = _hex_verts(centers[i], inner_rads[i])
    on = _outer_normals(outer_rad)
    P = (allv - c) @ on.T
    return P.max() / HALF_SQRT3


def _objective(params):
    """params: 2N centers + N inner angles + 1 outer angle (radians)."""
    centers = params[:2*N].reshape(N, 2)
    inner_rads = params[2*N:3*N]
    outer_rad = params[3*N]

    # Containment: max projection beyond inradius
    on = _outer_normals(outer_rad)
    allv = np.empty((N_INNER, 2))
    for i in range(N):
        allv[6*i:6*i+6] = _hex_verts(centers[i], inner_rads[i])
    P = allv @ on.T
    mx = P.max(axis=0)
    mn = P.min(axis=0)
    c = 0.5 * (mx + mn) @ on
    P2 = (allv - c) @ on.T
    contain_viol = P2.max() - HALF_SQRT3

    # Overlap penalty
    overlap = 0.0
    for i in range(N):
        v1 = allv[6*i:6*i+6]
        for j in range(i+1, N):
            v2 = allv[6*j:6*j+6]
            axes = np.vstack([_inner_normals(inner_rads[i]), _inner_normals(inner_rads[j])])
            d1 = v1 @ axes.T
            d2 = v2 @ axes.T
            gap = np.minimum(d1.max(axis=0), d2.max(axis=0)) - np.maximum(d1.min(axis=0), d2.min(axis=0))
            overlap += -gap.min()
    return contain_viol + 0.5 * overlap


def _nelder_mead(f, x0, max_iter=300, tol=1e-8, time_fn=None, deadline=None):
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = np.vstack([x0] + [x0 + 0.1*np.eye(n)[i] for i in range(n)])
    fvals = np.array([f(x) for x in simplex])
    for it in range(max_iter):
        if time_fn is not None and time_fn() > deadline:
            break
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha*(centroid - simplex[-1])
        fr = f(xr)
        if fr < fvals[0]:
            xe = centroid + gamma*(centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + rho*(simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for k in range(1, n):
                    simplex[k] = simplex[0] + sigma*(simplex[k] - simplex[0])
                    fvals[k] = f(simplex[k])
    idx = np.argmin(fvals)
    return simplex[idx], fvals[idx]


def construct():
    start = time.time()
    best_side = float('inf')
    best_params = None

    d1 = SQRT3
    first_ring = [(d1*np.cos(np.pi/3*i), d1*np.sin(np.pi/3*i)) for i in range(6)]
    second_ring = []
    for i in range(6):
        a = np.pi/3*i
        second_ring.append((2*d1*np.cos(a), 2*d1*np.sin(a)))
        a2 = np.pi/3*i + np.pi/6
        second_ring.append((2*d1*np.cos(a2), 2*d1*np.sin(a2)))

    # Build candidate templates: center + first ring + 4 from second ring
    templates = []
    for subset in combinations(range(12), 4):
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        # quick overlap check at angle 0
        ok = True
        for i in range(N):
            for j in range(i+1, N):
                dist = np.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
                if dist < 2*HALF_SQRT3 - 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            templates.append(centers)

    # Limit templates to keep within budget
    max_templates = 200
    templates = templates[:max_templates]

    for centers in templates:
        if time.time() > DEADLINE:
            break
        # Multi-start inner/outer angles
        for base_ang in (0.0, np.pi/6.0):
            if time.time() > DEADLINE:
                break
            inner_rads = np.full(N, base_ang)
            outer_rad = 0.0
            params = np.concatenate([
                np.array(centers).flatten(),
                inner_rads,
                [outer_rad]
            ])
            # Quick objective to skip obviously bad starts
            f0 = _objective(params)
            if f0 > 2.0:
                continue
            x, f = _nelder_mead(_objective, params, max_iter=250,
                                time_fn=time.time, deadline=DEADLINE)
            if time.time() > DEADLINE:
                break
            # Evaluate true side length
            centers_o = x[:2*N].reshape(N, 2)
            inner_rads_o = x[2*N:3*N]
            outer_rad_o = x[3*N]
            side, c = _outer_side_for(centers_o, inner_rads_o, outer_rad_o)
            if side < best_side - 1e-9:
                best_side = side
                best_params = (centers_o, inner_rads_o, outer_rad_o, c)

    # Fallback
    if best_params is None:
        centers = np.array([(2*d1*k, 0.0) for k in range(N)])
        inner_rads = np.zeros(N)
        outer_rad = 0.0
        side, c = _outer_side_for(centers, inner_rads, outer_rad)
        best_side = side
        best_params = (centers, inner_rads, outer_rad, c)

    centers, inner_rads, outer_rad, c = best_params
    inner = [[float(centers[i][0]), float(centers[i][1]), float(np.degrees(inner_rads[i]))] for i in range(N)]
    return inner, (float(c[0]), float(c[1])), float(best_side), float(np.degrees(outer_rad))
