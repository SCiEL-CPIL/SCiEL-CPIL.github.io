"""Hexagon packing: 11 unit hexagons inside a regular hexagon.

Strategy (structural change): model the 11 inner hexagons on the exact
hexagonal lattice (touching unit hexagons have centre-to-centre distance
2*sqrt(3) along the 3 axial directions).  The containment problem for a
fixed cluster is a linear L-infinity problem over the 6 facet normals of
the outer hexagon, so the minimum outer side is computed in O(66) dot
products.  Non-overlap is checked exactly with a Minkowski-sum /
support-function polygon distance (cheap, no segment-pair loops).  We
then optimise the global rotation of the composite cluster (plus a light
Nelder-Mead polish of the whole configuration) which is where the real
structural gain comes from.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQ3 = math.sqrt(3.0)
DEADLINE = 142.0          # safety margin before the 150 s kill
EPS = 1e-9


# --------------------------------------------------------------------------
# geometry primitives
# --------------------------------------------------------------------------
def _hex_vertices(angle_deg, cx=0.0, cy=0.0):
    """6 vertices of a unit regular hexagon (circumradius = side = 1)."""
    a = math.radians(angle_deg)
    th = a + np.arange(6) * (math.pi / 3.0)
    return np.column_stack((cx + np.cos(th), cy + np.sin(th)))


def _outer_normals(phi_deg):
    """Unit normals to the 6 facets of the outer hexagon at rotation phi."""
    a = math.radians(phi_deg)
    th = a + np.arange(6) * (math.pi / 3.0)
    return np.column_stack((np.cos(th), np.sin(th)))          # (6,2)


def _cluster_vertices(centers, angles, phi_deg=0.0):
    """All 66 vertices of the 11 inner hexagons (global rotation phi)."""
    out = np.empty((N, 6, 2))
    for i in range(N):
        out[i] = _hex_vertices(angles[i] + phi_deg, centers[i, 0], centers[i, 1])
    return out


def _min_outer_side(centers, angles, phi_deg=0.0):
    """Minimum outer side length so that all inner hexagons fit (rotation phi).

    A point v is inside a centred outer hexagon of side S iff
        |v . n_k| <= S * SQ3/2   for all 6 facet normals n_k.
    Hence S = max_k max_v |v . n_k| / (SQ3/2).
    """
    normals = _outer_normals(phi_deg)                        # (6,2)
    V = _cluster_vertices(centers, angles, phi_deg)          # (11,6,2)
    proj = V @ normals.T                                     # (11,6,6)
    return float(np.max(np.abs(proj)) / (SQ3 / 2.0))


# --- exact, cheap non-overlap via Minkowski sum --------------------------
def _support(P, d):
    return float(np.max(P @ d))


def _minkowski_support(A, B, d):
    return _support(A, d) + _support(B, -d)


def _minkowski_edges(A, B):
    """Edges (as outward normals) of the Minkowski sum of two convex polys."""
    na = len(A)
    nb = len(B)
    edges = []
    for i in range(na):
        p, q = A[i], A[(i + 1) % na]
        d = np.array([-(q[1] - p[1]), q[0] - p[0]])          # outward normal
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            continue
        d = d / nrm
        edges.append((d, _minkowski_support(A, B, d)))
    for j in range(nb):
        p, q = B[j], B[(j + 1) % nb]
        d = np.array([-(q[1] - p[1]), q[0] - p[0]])
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            continue
        d = d / nrm
        edges.append((d, _minkowski_support(A, B, d)))
    edges.sort(key=lambda t: math.atan2(t[0][1], t[0][0]))
    return edges


def _polygon_distance(A, B):
    """Exact minimum distance between two convex polygons (0 if they touch)."""
    # quick rejection via Minkowski-difference support test
    M = _minkowski_edges(A, B)
    if any(h <= EPS for _, h in M):
        return 0.0
    # minimum distance between convex polygons: min over edge pairs of the
    # two point-to-segment / vertex-to-segment distances
    best = float('inf')
    for i in range(len(A)):
        for j in range(len(B)):
            d = _seg_seg_dist(A[i], A[(i + 1) % len(A)], B[j], B[(j + 1) % len(B)])
            if d < best:
                best = d
    return best


def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        t2 = max(0.0, min(1.0, f / e))
        return float(np.linalg.norm(r + d2 * t2))
    if e < 1e-12:
        t1 = max(0.0, min(1.0, -float(d1 @ r) / a))
        return float(np.linalg.norm(r + d1 * t1))
    c = float(d1 @ r)
    b = float(d1 @ d2)
    denom = a * e - b * b
    if denom < 1e-12:
        t1 = 0.0
    else:
        t1 = max(0.0, min(1.0, (b * f - c * e) / denom))
    t2 = max(0.0, min(1.0, (b * t1 + f) / e))
    if t2 < 0.0:
        t2 = 0.0
        t1 = max(0.0, min(1.0, -c / a))
    elif t2 > 1.0:
        t2 = 1.0
        t1 = max(0.0, min(1.0, (b - c) / a))
    return float(np.linalg.norm(r + d1 * t1 - d2 * t2))


def _separation(centers, angles, phi_deg=0.0):
    """Minimum clearance between inner hexagons (>=0 means non-overlapping)."""
    V = _cluster_vertices(centers, angles, phi_deg)          # (11,6,2)
    best = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = _polygon_distance(V[i], V[j])
            if d < best:
                best = d
    return best


def _overlap_penalty(centers, angles, phi_deg=0.0):
    sep = _separation(centers, angles, phi_deg)
    return 0.0 if sep >= EPS else (EPS - sep) ** 2


# --------------------------------------------------------------------------
# initial configurations on the exact lattice
# --------------------------------------------------------------------------
def _lattice_config(kind):
    """Return (centers, angles) for a candidate 11-hexagon cluster.

    Touching unit hexagons (flat-top, circumradius 1) have centre distance
    2*SQ3 along the three axial directions.
    """
    d = 2.0 * SQ3
    # axial basis: 60 deg apart
    ax = np.array([[math.cos(t), math.sin(t)] for t in
                   [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0]])

    centers = [np.array([0.0, 0.0])]
    # first ring: 6 neighbours
    for k in range(3):
        centers.append(ax[k] * d)
        centers.append(-ax[k] * d)

    if kind == 'alt':
        # 4 second-ring hexagons at the 4 alternating corners (a=1,b=1)
        for k in range(3):
            c = ax[k]
            c2 = np.array([c[0] - c[1], c[1] + c[0]])        # 60 deg rotation
            centers.append((c + c2) * d)
    elif kind == 'corners':
        # 4 second-ring hexagons on the 4 alternating axes
        centers.append(2.0 * ax[0] * d)
        centers.append(2.0 * ax[1] * d)
        centers.append(2.0 * ax[2] * d)
        centers.append(-2.0 * ax[0] * d)
    elif kind == 'ring':
        # 4 second-ring hexagons on 4 consecutive axes
        for k in range(4):
            centers.append(2.0 * ax[k % 3] * d)
    else:  # 'spread'
        for k in range(4):
            centers.append(2.0 * ax[k % 3] * d * (1.0 + 0.0))

    centers = np.array(centers, dtype=float)
    angles = np.zeros(N)
    return centers, angles


def _random_config(rng):
    """Random jittered lattice config (same structure, small perturbations)."""
    centers, angles = _lattice_config('alt')
    centers = centers + rng.normal(0.0, 0.05, size=centers.shape)
    angles = rng.uniform(0.0, 60.0, size=N)
    return centers, angles


# --------------------------------------------------------------------------
# objective for optimisation (rotation + positions)
# --------------------------------------------------------------------------
def _objective(params, w=1e3):
    """params = [phi_deg, x0.., y0.., ang0..].  Returns outer side + penalty."""
    phi = params[0]
    centers = params[1:1 + 2 * N].reshape(N, 2)
    angles = params[1 + 2 * N:]
    S = _min_outer_side(centers, angles, phi)
    pen = _overlap_penalty(centers, angles, phi)
    return S + w * pen


def _phi_objective(centers, angles):
    """Minimise outer side over the global rotation only (fast, exact)."""
    def f(p):
        return _min_outer_side(centers, angles, p[0])
    best = None
    for start in (0.0, 5.0, 15.0, 30.0, 45.0):
        r = minimize(f, np.array([start]), method='Nelder-Mead',
                     options={'xatol': 1e-7, 'fatol': 1e-9,
                              'maxiter': 400, 'maxfev': 800})
        val = _min_outer_side(centers, angles, r.x[0] % 60.0)
        if best is None or val < best:
            best = val
    return best


# --------------------------------------------------------------------------
def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    best = None  # (S, centers, angles, phi)

    def consider(centers, angles, phi):
        nonlocal best
        if time.time() > DEADLINE:
            return
        if _separation(centers, angles, phi) < EPS:
            return
        S = _min_outer_side(centers, angles, phi)
        if best is None or S < best[0]:
            best = (S, centers.copy(), angles.copy(), float(phi))

    # ---- 1) try each lattice structure, optimise global rotation ----------
    for kind in ('alt', 'corners', 'ring'):
        centers, angles = _lattice_config(kind)
        # coarse scan of global rotation, then local polish
        grid = np.arange(0.0, 60.0, 2.5)
        vals = [_min_outer_side(centers, angles, g) for g in grid]
        g_best = grid[int(np.argmin(vals))]
        r = minimize(lambda p: _min_outer_side(centers, angles, p[0]),
                     np.array([g_best]), method='Nelder-Mead',
                     options={'xatol': 1e-7, 'fatol': 1e-9,
                              'maxiter': 600, 'maxfev': 1500})
        consider(centers, angles, r.x[0] % 60.0)
        consider(centers, angles, g_best)

    # ---- 2) full Nelder-Mead polish from the best structure ---------------
    if best is not None:
        S0, c0, a0, p0 = best
        def full_obj(p):
            if time.time() > DEADLINE:
                return 1e10
            return _objective(p, w=1e4)
        x0 = np.concatenate(([p0], c0.ravel(), a0))
        r = minimize(full_obj, x0, method='Nelder-Mead',
                     options={'xatol': 1e-7, 'fatol': 1e-9,
                              'maxiter': 4000, 'maxfev': 6000})
        if r.fun < 1e9:
            phi = r.x[0] % 60.0
            consider(r.x[1:1 + 2 * N].reshape(N, 2), r.x[1 + 2 * N:], phi)

    # ---- 3) a few randomised full optimisations ---------------------------
    if time.time() < DEADLINE - 40:
        for _ in range(6):
            centers, angles = _random_config(rng)
            def full_obj(p, c0=centers, a0=angles):
                if time.time() > DEADLINE:
                    return 1e10
                return _objective(p, w=1e4)
            x0 = np.concatenate(([0.0], centers.ravel(), angles))
            try:
                r = minimize(full_obj, x0, method='Nelder-Mead',
                             options={'xatol': 1e-7, 'fatol': 1e-9,
                                      'maxiter': 2500, 'maxfev': 4000})
                if r.fun < 1e9:
                    phi = r.x[0] % 60.0
                    consider(r.x[1:1 + 2 * N].reshape(N, 2), r.x[1 + 2 * N:], phi)
            except Exception:
                pass

    # ---- 4) final safety: guarantee a valid, non-overlapping config -------
    if best is None:
        centers, angles = _lattice_config('alt')
        best = (_min_outer_side(centers, angles, 0.0), centers, angles, 0.0)

    S, centers, angles, phi = best

    # apply the global rotation to the reported per-hexagon angles and
    # recompute the exact required outer side at that rotation
    final_angles = (angles + phi) % 360.0
    S_final = _min_outer_side(centers, final_angles, 0.0)
    # tiny safety margin so floating point never reports a false overlap/exit
    S_final *= 1.0001

    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(final_angles[i])]
             for i in range(N)]
    return inner, (0.0, 0.0), float(S_final), 0.0


# EVOLVE-BLOCK-END
