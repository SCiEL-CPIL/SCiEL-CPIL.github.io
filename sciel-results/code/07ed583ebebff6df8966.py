# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before 600s/630s kill


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _min_dist_convex(P, Q):
    """
    Calculate the minimum distance between two convex polygons P and Q.
    Returns 0 if they intersect or touch.
    P, Q are (6, 2) arrays of vertices.
    """
    # If polygons intersect, distance is 0.
    # We can check intersection via SAT: if any axis separates them, no intersection.
    # If no axis separates them, they intersect (or one is inside the other, which is an intersection for our purposes of 'gap').
    # Actually, simpler: if they intersect, min dist is 0.
    # We check all normals from P and Q.
    
    def get_normals(poly):
        norms = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal /= np.linalg.norm(normal)
            # Ensure normal points outward
            # Centroid
            c = np.mean(poly, axis=0)
            mid = (p1 + p2) / 2
            if np.dot(normal, mid - c) < 0:
                normal = -normal
            norms.append(normal)
        return norms

    norms_P = get_normals(P)
    norms_Q = get_normals(Q)
    all_norms = norms_P + norms_Q
    
    min_dist = float('inf')
    
    for n in all_norms:
        proj_P = P @ n
        proj_Q = Q @ n
        
        min_P = proj_P.min()
        max_P = proj_P.max()
        min_Q = proj_Q.min()
        max_Q = proj_Q.max()
        
        # Check for separation
        if max_P < min_Q:
            dist = min_Q - max_P
        elif max_Q < min_P:
            dist = min_P - max_Q
        else:
            # Overlapping on this axis
            dist = 0.0
        
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def _separation(h1, h2):
    """Return min gap between two unit hexagons (>=0 if disjoint)."""
    # h1, h2 are (x, y, angle)
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    return _min_dist_convex(v1, v2)

def _inner_outside_violation(inner, outer):
    """Sum of how far inner vertices lie outside the outer hexagon (0 if all inside)."""
    # outer is (x, y, angle, side)
    ov = _hex_vertices(outer[0], outer[1], outer[3], outer[2])
    n = len(ov)
    viol = 0.0
    for h in inner:
        v = _hex_vertices(h[0], h[1], 1.0, h[2])
        for p in v:
            for i in range(n):
                p1 = ov[i]
                p2 = ov[(i + 1) % n]
                # Edge vector
                e = p2 - p1
                # Normal pointing inward? 
                # Let's use cross product to check side.
                # For a CCW polygon, cross(edge, point - p1) > 0 means point is to the left (inside).
                # Our hexagon vertices are generated CCW?
                # cos/sin with increasing angle is CCW.
                # So cross(e, p - p1) should be >= 0 for inside.
                cross = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if cross < 0:
                    # Distance to line
                    dist = -cross / np.linalg.norm(e)
                    viol += dist
    return viol

def _outer_side_for(inner):
    """Min outer side (centered at origin, angle 0) such that all inner fit inside."""
    # Optimization: The outer hexagon is centered at (0,0) and has angle 0.
    # We can binary search on side length S.
    lo, hi = 3.0, 30.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        outer = np.array([0.0, 0.0, 0.0, mid])
        if _inner_outside_violation(inner, outer) < 1e-9:
            hi = mid
        else:
            lo = mid
    return hi

def _objective(x, n=11):
    inner = x.reshape(n, 3)
    # penalty for overlaps
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep = _separation(inner[i], inner[j])
            if sep < 1e-6:
                pen += (1e-6 - sep) ** 2 * 100.0
    side = _outer_side_for(inner)
    return side + pen


def hexagon_packing_11():
    n = 11
    # Good initial grid packing (seed)
    inner0 = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
    ])

    best_x = inner0.copy()
    best_side = _outer_side_for(inner0)

    # Try a few local optimizations with different starts
    starts = [inner0]
    # jittered starts
    rng = np.random.default_rng(0)
    for _ in range(5):
        starts.append(inner0 + rng.normal(0, 0.5, inner0.shape))

    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break
        try:
            res = minimize(_objective, s0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-6})
            x = res.x.reshape(n, 3)
            # check validity of separation
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    if _separation(x[i], x[j]) < 1e-6:
                        ok = False
            if ok:
                side = _outer_side_for(x)
                if side < best_side:
                    best_side = side
                    best_x = x.copy()
        except Exception:
            continue

    inner = best_x
    side = best_side
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
