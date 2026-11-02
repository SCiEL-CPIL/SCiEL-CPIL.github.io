"""Packing 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    # regular hexagon vertices at angles a + k*60deg, radius = side
    ks = np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(a + ks),
                            cy + side * np.sin(a + ks)])

def _hexagon_support(side, center, rot_deg):
    """Return (normals, offset) such that constraint is max_k (n_k . (p - center)) <= side*sqrt(3)/2.
    normals are unit vectors at angles rot_deg + k*60."""
    a = np.deg2rad(rot_deg)
    ks = np.arange(6) * np.pi / 3.0
    normals = np.column_stack([np.cos(a + ks), np.sin(a + ks)])
    return normals

def _enclosing_side(points, center, rot_deg):
    """Minimum side length of a regular hexagon (center, rot) enclosing all points."""
    normals = _hexagon_support(1.0, center, rot_deg)  # just need directions
    rel = points - np.array(center)
    # max projection in each of the 6 normal directions
    proj = rel @ normals.T  # shape (npts, 6)
    maxproj = proj.max(axis=0)  # (6,)
    # For enclosing, we need maxproj_k <= side*sqrt(3)/2 for all k
    # but also the hexagon is symmetric: the constraint is that for each normal n_k,
    # the max of n_k.(p-c) over p must be <= inradius = side*sqrt(3)/2
    # However the hexagon of side s has inradius s*sqrt(3)/2, and the half-plane
    # in direction n_k is at distance s*sqrt(3)/2. So:
    m = maxproj.max()
    return (2.0 / np.sqrt(3.0)) * m

def construct():
    t0 = time.time()
    deadline = 120.0  # safety margin before 150s kill

    # Build a rigid cluster of 12 unit hexagons in a compact arrangement.
    # Use a "flower" base: center + 6 neighbors (7 hexes), plus 5 more.
    # Positions of hex centers for a honeycomb of side-1 hexagons:
    # neighbor distance between adjacent hex centers = sqrt(3) (for side 1, center-to-center = 2*inradius = sqrt(3)).
    d = np.sqrt(3.0)
    # base 7-hex flower centers (relative)
    base = [(0.0, 0.0)]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        base.append((d * np.cos(ang), d * np.sin(ang)))
    # add 5 more around the ring (a 19-hex flower would add 12; we add 5 of those)
    # 19-hex flower ring-2 centers at distance 2d in directions 60k and 60k+30
    ring2 = []
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        ring2.append((2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang)))
        ang2 = np.deg2rad(60.0 * k + 30.0)
        ring2.append((2.0 * d * np.cos(ang2), 2.0 * d * np.sin(ang2)))
    extra = ring2[:5]
    centers_rel = np.array(base + extra)  # 12 centers
    # Each inner hexagon has the same orientation as the cluster (angle 0 relative)
    # Compute the set of all vertices of all 12 hexagons (for enclosing computation)
    all_pts = []
    for (cx, cy) in centers_rel:
        all_pts.append(_hex_vertices(cx, cy, 0.0))
    pts = np.vstack(all_pts)  # (72, 2)

    best = None
    best_side = float('inf')

    def evaluate(cx, cy, rot_deg):
        # translate and rotate pts by cluster transform
        a = np.deg2rad(rot_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p = pts @ R.T + np.array([cx, cy])
        # enclosing hexagon centered at (cx,cy) with same rotation rot_deg
        normals = _hexagon_support(1.0, (cx, cy), rot_deg)
        rel = p - np.array([cx, cy])
        proj = rel @ normals.T
        m = proj.max(axis=0).max()
        return (2.0 / np.sqrt(3.0)) * m

    # Coarse grid search over rotation (outer hexagon rotation relative to cluster)
    # and center offset. By symmetry we can fix cluster at origin and vary outer hex rot & center.
    # Actually the cluster position relative to outer center matters; but we can fix outer center at origin
    # and let cluster center be (cx,cy). Equivalent to fixing cluster center at origin and moving outer hex.
    # Simpler: fix cluster centered at origin; optimize outer hex center (cx,cy) and rotation.
    # Re-express: enclosing side of pts (cluster at origin) by hex(center=(cx,cy), rot):
    def eval2(cx, cy, rot_deg):
        normals = _hexagon_support(1.0, (cx, cy), rot_deg)
        rel = pts - np.array([cx, cy])
        proj = rel @ normals.T
        m = proj.max(axis=0).max()
        return (2.0 / np.sqrt(3.0)) * m

    # coarse search over rot (outer hex rotation) and center offset
    for rot in np.linspace(0.0, 60.0, 61):
        # center offset: search over a small grid; by symmetry offset magnitude small
        for cx in np.linspace(-1.0, 1.0, 21):
            for cy in np.linspace(-1.0, 1.0, 21):
                s = eval2(cx, cy, rot)
                if s < best_side:
                    best_side = s
                    best = (cx, cy, rot)
        if time.time() - t0 > deadline:
            break

    # local polish with Nelder-Mead on (cx, cy, rot)
    if best is not None and time.time() - t0 < deadline:
        from scipy.optimize import minimize
        x0 = np.array([best[0], best[1], best[2]])
        res = minimize(lambda x: eval2(x[0], x[1], x[2]), x0,
                       method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 4000, 'maxfev': 4000})
        if res.fun < best_side:
            best_side = float(res.fun)
            best = (float(res.x[0]), float(res.x[1]), float(res.x[2]))

    # Also try rotating the cluster itself (inner hexes share orientation) — but cluster shape is fixed;
    # rotating the cluster is equivalent to rotating the whole configuration, which is captured by
    # varying outer hex rotation. So no extra needed.

    # Build final inner list: each hex center = cluster center (0,0) + rotated relative center
    cx, cy, rot = best
    a = np.deg2rad(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    inner = []
    for (rx, ry) in centers_rel:
        px, py = R @ np.array([rx, ry])
        # inner hex orientation = rot (same as cluster)
        inner.append([float(px), float(py), float(rot)])

    outer_center = (float(cx), float(cy))
    side = float(best_side)
    angle = float(rot)
    return inner, outer_center, side, angle
