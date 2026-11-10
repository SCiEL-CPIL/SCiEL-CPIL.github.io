# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + d * np.cos(ang), cy + d * np.sin(ang)], axis=1)


def _hex_edge_normals(V):
    # 6 edge outward normals (unit)
    n = np.empty((6, 2))
    for i in range(6):
        p1 = V[i]
        p2 = V[(i + 1) % 6]
        e = p2 - p1
        L = np.hypot(e[0], e[1])
        e = e / L
        n[i] = np.array([e[1], -e[0]])
    return n


def _separation_ok(V1, N1, V2, N2, tol=1e-6):
    # SAT: if any axis has gap > tol, separated
    for N in (N1, N2):
        for axis in N:
            a1 = V1 @ axis
            a2 = V2 @ axis
            if a1.max() < a2.min() - tol or a2.max() < a1.min() - tol:
                return True
    return False


def _min_separation(V1, N1, V2, N2):
    # largest gap over all axes; positive => separated
    best = -np.inf
    for N in (N1, N2):
        for axis in N:
            a1 = V1 @ axis
            a2 = V2 @ axis
            gap = max(a1.max() - a2.min(), a2.max() - a1.min())
            if gap > best:
                best = gap
    return best


def _contains(Vouter, p, tol=1e-6):
    # p inside/on outer hexagon (CCW)
    for i in range(6):
        p1 = Vouter[i]
        p2 = Vouter[(i + 1) % 6]
        if np.cross(p2 - p1, p - p1) < -tol:
            return False
    return True


def _build(theta, phi, s):
    # theta[12], phi[12]: radial distances and angle deviations (deg)
    # ring angles for the 11 ring hexes (center has angle 0)
    ring_angles = np.deg2rad(60.0 * np.arange(11))
    centers = np.empty((12, 2))
    centers[0] = [0.0, 0.0]
    for i in range(1, 12):
        a = ring_angles[i - 1] + np.deg2rad(phi[i])
        centers[i] = [theta[i] * np.cos(a), theta[i] * np.sin(a)]
    inner = np.zeros((12, 3))
    inner[:, 0] = centers[:, 0]
    inner[:, 1] = centers[:, 1]
    inner[:, 2] = 0.0
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, s


def _outer_from_centers(centers, margin=0.0):
    # tight regular hexagon (centered at origin) containing all vertices
    verts = np.empty((12, 6, 2))
    for i in range(12):
        verts[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
    allv = verts.reshape(-1, 2)
    # regular hexagon centered at origin, side s, angle 0:
    # vertex k at (s cos 60k, s sin 60k). Edge k from vertex k to k+1.
    # outward normal of edge k is at angle 60k + 30.
    ang = np.deg2rad(60.0 * np.arange(6) + 30.0)
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (6,2)
    # support of allv in each normal direction
    sup = allv @ normals.T  # (12*6, 6)
    maxsup = sup.max(axis=0)  # (6,)
    # edge k line: dot(normal_k, x) = s * cos(30deg)
    s_needed = maxsup / np.cos(np.deg2rad(30.0))
    s = float(s_needed.max()) + margin
    return s


def _feasibility(theta, phi):
    # returns (min_sep, min_contain) both should be >= 0
    centers = np.empty((12, 2))
    ring_angles = np.deg2rad(60.0 * np.arange(11))
    centers[0] = [0.0, 0.0]
    for i in range(1, 12):
        a = ring_angles[i - 1] + np.deg2rad(phi[i])
        centers[i] = [theta[i] * np.cos(a), theta[i] * np.sin(a)]
    V = np.empty((12, 6, 2))
    N = np.empty((12, 6, 2))
    for i in range(12):
        V[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
        N[i] = _hex_edge_normals(V[i])
    minsep = np.inf
    for i in range(12):
        for j in range(i + 1, 12):
            g = _min_separation(V[i], N[i], V[j], N[j])
            if g < minsep:
                minsep = g
    # containment: outer hexagon from centers
    s = _outer_from_centers(centers, margin=0.0)
    Vouter = _hex_vertices(0.0, 0.0, s, 0.0)
    mincont = np.inf
    for i in range(12):
        for p in V[i]:
            # distance to outside: for each edge, cross >= 0 means inside
            worst = 0.0
            for k in range(6):
                p1 = Vouter[k]
                p2 = Vouter[(k + 1) % 6]
                c = np.cross(p2 - p1, p - p1)
                if c < worst:
                    worst = c
            if worst < mincont:
                mincont = worst
    return minsep, mincont


def _objective(x, margin):
    # x = [theta(12), phi(12)]
    theta = x[:12]
    phi = x[12:]
    centers = np.empty((12, 2))
    ring_angles = np.deg2rad(60.0 * np.arange(11))
    centers[0] = [0.0, 0.0]
    for i in range(1, 12):
        a = ring_angles[i - 1] + np.deg2rad(phi[i])
        centers[i] = [theta[i] * np.cos(a), theta[i] * np.sin(a)]
    s = _outer_from_centers(centers, margin=margin)
    # constraints handled via penalty
    minsep, mincont = _feasibility(theta, phi)
    pen = 0.0
    if minsep < 0:
        pen += 1000.0 * (-minsep)
    if mincont < 0:
        pen += 1000.0 * (-mincont)
    return s + pen


def hexagon_packing_12():
    deadline = time.time() + 600.0
    # ideal honeycomb: center + 11 ring at radius 2*sqrt(3)
    r0 = 2.0 * np.sqrt(3.0)
    theta0 = np.full(12, r0)
    theta0[0] = 0.0
    phi0 = np.zeros(12)
    x0 = np.concatenate([theta0, phi0])

    best_x = x0.copy()
    best_s = _outer_from_centers(np.zeros((12, 2)), 0.0)
    # compute initial s
    centers0 = np.empty((12, 2))
    ring_angles = np.deg2rad(60.0 * np.arange(11))
    centers0[0] = [0.0, 0.0]
    for i in range(1, 12):
        a = ring_angles[i - 1]
        centers0[i] = [r0 * np.cos(a), r0 * np.sin(a)]
    best_s = _outer_from_centers(centers0, 0.0)
    best_minsep, best_mincont = _feasibility(theta0, phi0)

    margin = 0.02
    bounds = [(0.0, 10.0) for _ in range(12)] + [(-30.0, 30.0) for _ in range(12)]
    # fix center radius small
    bounds[0] = (0.0, 0.5)

    methods = ["L-BFGS-B", "Nelder-Mead", "Powell"]
    for method in methods:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                _objective,
                x0,
                method=method,
                bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-8},
            )
        except Exception:
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        theta = res.x[:12]
        phi = res.x[12:]
        minsep, mincont = _feasibility(theta, phi)
        if minsep < -1e-5 or mincont < -1e-5:
            continue
        s = _outer_from_centers(
            np.empty((12, 2)) and _centers_from(theta, phi), 0.0
        )
        if s < best_s:
            best_s = s
            best_x = res.x.copy()
            best_minsep, best_mincont = minsep, mincont
            # update x0 to continue from best
            x0 = res.x.copy()

    # final: ensure feasibility with a small safety margin on s
    theta = best_x[:12]
    phi = best_x[12:]
    centers = _centers_from(theta, phi)
    s = _outer_from_centers(centers, margin=1e-4)
    inner, outer, _ = _build(theta, phi, s)
    return inner, outer, s


def _centers_from(theta, phi):
    centers = np.empty((12, 2))
    ring_angles = np.deg2rad(60.0 * np.arange(11))
    centers[0] = [0.0, 0.0]
    for i in range(1, 12):
        a = ring_angles[i - 1] + np.deg2rad(phi[i])
        centers[i] = [theta[i] * np.cos(a), theta[i] * np.sin(a)]
    return centers


# EVOLVE-BLOCK-END
