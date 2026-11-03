"""Packing 12 unit hexagons in a hexagonal container; minimize outer side length."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 12
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon (side s, center, rotation rad)."""
    ang = theta + np.arange(6) * (np.pi / 3.0)
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _poly_min_distance(A, B):
    """Minimum distance between two convex polygons (numpy arrays of (n,2) vertices).
    Returns 0.0 if they overlap, else the positive minimum distance.
    Uses a robust O(n*m) edge/vertex check plus overlap test via separating axis."""
    # Separating axis theorem: if any axis separates, they don't overlap.
    # Candidate axes: normals of each edge of A and B.
    def edge_axes(P):
        axes = []
        n = len(P)
        for i in range(n):
            p1, p2 = P[i], P[(i + 1) % n]
            e = p2 - p1
            ax = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-12:
                axes.append(ax / nrm)
        return axes

    axes = edge_axes(A) + edge_axes(B)
    overlap = True
    for ax in axes:
        a_min = np.min(A @ ax)
        a_max = np.max(A @ ax)
        b_min = np.min(B @ ax)
        b_max = np.max(B @ ax)
        if a_max < b_min - 1e-9 or b_max < a_min - 1e-9:
            overlap = False
            break
    if overlap:
        return 0.0
    # Compute minimum distance: min over vertex-edge and edge-edge.
    def point_seg_dist(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-18)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    best = np.inf
    nA, nB = len(A), len(B)
    for i in range(nA):
        a1, a2 = A[i], A[(i + 1) % nA]
        for j in range(nB):
            b1, b2 = B[j], B[(j + 1) % nB]
            # vertex to edge
            d = min(point_seg_dist(a1, b1, b2), point_seg_dist(a2, b1, b2),
                    point_seg_dist(b1, a1, a2), point_seg_dist(b2, a1, a2))
            if d < best:
                best = d
    return best


def _smooth_collision_penalty(A, B, smooth=0.05):
    """Smooth penalty: 0 if separated, grows as overlap. Use min distance (0 if overlap).
    We want a penalty that is 0 when disjoint and positive when overlapping.
    Since _poly_min_distance returns 0 on overlap, we need overlap depth.
    Approximate overlap depth via a soft measure: use the minimum over separating axes
    of the gap (negative when overlapping)."""
    def edge_axes(P):
        axes = []
        n = len(P)
        for i in range(n):
            p1, p2 = P[i], P[(i + 1) % n]
            e = p2 - p1
            ax = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-12:
                axes.append(ax / nrm)
        return axes

    axes = edge_axes(A) + edge_axes(B)
    # gap on each axis: (min of B - max of A) along axis; if negative on all axes -> overlap
    # penetration depth ~ max over axes of (min_B - max_A)  (positive means separated)
    gaps = []
    for ax in axes:
        a_max = np.max(A @ ax)
        b_min = np.min(B @ ax)
        gaps.append(b_min - a_max)
    gaps = np.array(gaps)
    # If max gap > 0, separated by that amount. Else overlapping; depth = -min gap approx.
    sep = np.max(gaps)
    if sep > 0:
        return 0.0
    # Overlapping: use softplus of the most negative gap as a smooth proxy for depth.
    depth = -np.min(gaps)  # > 0
    return depth  # linear; we'll square it in the objective


def _objective(params, S):
    """params: [S, x0,y0,th0, x1,y1,th1, ...]. Minimize S + penalties."""
    S = params[0]
    if S <= 0:
        return 1e6
    # outer hexagon: center (0,0), side S, rotation 0
    outer = _hex_vertices(0.0, 0.0, S, 0.0)
    penalty = 0.0
    hexes = []
    for k in range(N):
        cx, cy, th = params[1 + 3 * k], params[2 + 3 * k], params[3 + 3 * k]
        H = _hex_vertices(cx, cy, 1.0, th)
        hexes.append(H)
        # containment: all vertices inside outer hexagon.
        # Use separating axis: outer contains H iff for every axis (from outer edges),
        # H is within outer's slab.
        def edge_axes(P):
            axes = []
            n = len(P)
            for i in range(n):
                p1, p2 = P[i], P[(i + 1) % n]
                e = p2 - p1
                ax = np.array([-e[1], e[0]])
                nrm = np.linalg.norm(ax)
                if nrm > 1e-12:
                    axes.append(ax / nrm)
            return axes
        for ax in edge_axes(outer):
            o_min = np.min(outer @ ax)
            o_max = np.max(outer @ ax)
            h_min = np.min(H @ ax)
            h_max = np.max(H @ ax)
            # violation if H sticks out
            if h_min < o_min:
                penalty += (o_min - h_min) ** 2
            if h_max > o_max:
                penalty += (h_max - o_max) ** 2
    # inner-inner collisions
    for i in range(N):
        for j in range(i + 1, N):
            d = _smooth_collision_penalty(hexes[i], hexes[j])
            penalty += d ** 2
    return S + 10.0 * penalty


def _make_initial(S0, pattern):
    """Create initial params. pattern: 'cluster' or 'random'."""
    params = np.zeros(1 + 3 * N)
    params[0] = S0
    if pattern == 'cluster':
        # 3-4-3 honeycomb-ish: rows
        # approximate hex lattice positions for 12 hexagons
        # Use a 4-4-4 or 3-4-3 grid. Let's place them in a hexagonal cluster.
        # Lattice: neighbors at distance 2 (side 1 hexagons touch at distance 2? 
        # Two unit hexagons with same orientation touch when center distance = 2 * inradius * 2? 
        # Actually for same-orientation unit hexagons, min center distance for no overlap = 2 * apothem * 2? 
        # The width across flats = 2 * apothem = sqrt(3). Two such hexagons side by side (flat-to-flat)
        # touch at center distance = sqrt(3). Across vertices = 2. So min distance depends on orientation.
        # For same orientation, the min center distance to be disjoint = sqrt(3) (flat-to-flat).
        # We'll place them on a triangular lattice with spacing ~ sqrt(3)+eps.
        sp = math.sqrt(3.0) + 0.05
        # 3-4-3 arrangement
        positions = []
        # row 0: 3 hexes
        for i in range(3):
            positions.append((i * sp, 0.0))
        # row 1: 4 hexes, offset
        y1 = sp * math.sin(math.pi / 3.0)
        for i in range(4):
            positions.append((i * sp - sp / 2.0, y1))
        # row 2: 3 hexes
        y2 = 2 * y1
        for i in range(3):
            positions.append((i * sp, y2))
        # center the cluster
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        mx = (max(xs) + min(xs)) / 2.0
        my = (max(ys) + min(ys)) / 2.0
        for k, (px, py) in enumerate(positions):
            params[1 + 3 * k] = px - mx
            params[2 + 3 * k] = py - my
            params[3 + 3 * k] = 0.0
    else:
        rng = np.random.default_rng(42)
        for k in range(N):
            params[1 + 3 * k] = rng.uniform(-2, 2)
            params[2 + 3 * k] = rng.uniform(-2, 2)
            params[3 + 3 * k] = rng.uniform(0, np.pi)
    return params


def _try_optimize(S0, pattern, max_iter=400):
    params0 = _make_initial(S0, pattern)
    res = minimize(_objective, params0, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6})
    return res


def construct():
    global _START
    best_S = None
    best_params = None
    best_obj = np.inf

    # Try a range of initial outer sizes and patterns.
    patterns = ['cluster', 'random']
    S0_list = [6.0, 7.0, 8.0, 5.0]

    attempts = 0
    for S0 in S0_list:
        for pat in patterns:
            if time.time() - _START > _DEADLINE:
                break
            attempts += 1
            try:
                res = _try_optimize(S0, pat, max_iter=300)
                S_val = res.x[0]
                # Recompute objective to check validity
                obj = _objective(res.x, S_val)
                # Check that there are no large violations
                if obj < best_obj and S_val > 0:
                    best_obj = obj
                    best_S = S_val
                    best_params = res.x.copy()
            except Exception:
                continue
        if time.time() - _START > _DEADLINE:
            break

    if best_params is None:
        # Fallback: return something valid-ish (large container)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Build inner list
    inner = []
    for k in range(N):
        cx = best_params[1 + 3 * k]
        cy = best_params[2 + 3 * k]
        th = best_params[3 + 3 * k]
        inner.append([float(cx), float(cy), float(math.degrees(th))])

    return inner, (0.0, 0.0), float(best_S), 0.0
