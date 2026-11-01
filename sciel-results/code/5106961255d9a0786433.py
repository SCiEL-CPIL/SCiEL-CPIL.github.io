"""Rigid-body cluster approach: fixed honeycomb arrangement, optimize global transform + outer hex."""
import math
import time
import numpy as np

N = 12
DEADLINE = 135.0  # safety margin under 150s kill

# ---- Honeycomb cluster: 7 hex in a flower + 5 around the top ring ----
# Unit hexagon: circumradius = 1, inradius = sqrt(3)/2.
# Neighbour centres for a regular honeycomb (pointy-top, 60deg steps) are at distance 2*inradius = sqrt(3).
# Use flat-top orientation: neighbours at angles 0,60,120,180,240,300 deg, distance sqrt(3).
#
# Coordinates (x right, y up), centre at origin:
#   Flower (7):
#     (0,0), and 6 at angle k*60, r=sqrt(3)
#   Top ring (5): the 5 neighbours of the top three hexes that are not already in the flower.
#
# Let's build it programmatically to guarantee validity.

def _build_cluster():
    s3 = math.sqrt(3.0)
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60.0 * k)
        pts.append((s3 * math.cos(a), s3 * math.sin(a)))
    # Now add a second ring around the top 3 hexes (angles 0,60,120 -> actually top is 90deg; use angles 60,120,180? )
    # Simpler: add 5 more at distance 2*sqrt(3) from origin at angles 90, 30, 150, 210, 330? Let's just add ring-2 neighbours.
    # Add ring-2: for each of the 6 ring-1 hexes, add its outward neighbour (distance sqrt(3) further out) at same angle.
    # That gives 6 more -> total 13. We need 12, so drop one.
    for k in range(6):
        a = math.radians(60.0 * k)
        pts.append((2.0 * s3 * math.cos(a), 2.0 * s3 * math.sin(a)))
    # We have 13; drop the last one (angle 300) to keep 12.
    pts = pts[:12]
    return np.array(pts, dtype=float)

_CLUSTER = _build_cluster()

def _hex_vertices(cx, cy, rot_deg):
    """Vertices of a unit regular hexagon (circumradius 1) centred at (cx,cy), rotated by rot_deg."""
    r = math.radians(rot_deg)
    verts = []
    for k in range(6):
        a = r + math.radians(60.0 * k)
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return np.array(verts)

def _outer_hex_vertices(cx, cy, side, rot_deg):
    """Vertices of regular hexagon with side length `side`, centre (cx,cy), rotation rot_deg.
    Circumradius = side."""
    r = math.radians(rot_deg)
    verts = []
    for k in range(6):
        a = r + math.radians(60.0 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def _point_in_convex_polygon(p, poly):
    """Return True if point p is strictly inside (or on) convex polygon poly (CCW)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        # cross product of (b-a) x (p-a); for CCW polygon, inside means >= 0 for all edges
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -1e-9:
            return False
    return True

def _all_inside(inner_centers, inner_rot, outer_cx, outer_cy, outer_side, outer_rot):
    """Check every vertex of every inner hex is inside the outer hex."""
    outer = _outer_hex_vertices(outer_cx, outer_cy, outer_side, outer_rot)
    for (cx, cy) in inner_centers:
        verts = _hex_vertices(cx, cy, inner_rot)
        for v in verts:
            if not _point_in_convex_polygon(v, outer):
                return False
    return True

def _score_candidate(tx, ty, rot_deg, outer_cx, outer_cy, outer_side, outer_rot):
    """Return outer_side if feasible, else inf."""
    centers = _CLUSTER + np.array([tx, ty])
    if not _all_inside(centers, rot_deg, outer_cx, outer_cy, outer_side, outer_rot):
        return math.inf
    return outer_side

# For speed, precompute cluster vertices in local frame (rotation 0), then rotate+translate as a rigid body.
def _precompute_local_vertices():
    """Local vertices of all 12 hexes (rotation 0), as (72,2)."""
    allv = []
    for (cx, cy) in _CLUSTER:
        for k in range(6):
            a = math.radians(60.0 * k)
            allv.append((cx + math.cos(a), cy + math.sin(a)))
    return np.array(allv, dtype=float)

_LOCAL_VERTS = _precompute_local_vertices()  # shape (72,2)

def _rigid_transform(verts, tx, ty, rot_deg):
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c, -s], [s, c]])
    return verts @ R.T + np.array([tx, ty])

def _outer_edges(outer_cx, outer_cy, outer_side, outer_rot):
    """Return list of (a,b) edge endpoints for the outer hex (CCW)."""
    r = math.radians(outer_rot)
    verts = []
    for k in range(6):
        a = r + math.radians(60.0 * k)
        verts.append((outer_cx + outer_side * math.cos(a), outer_cy + outer_side * math.sin(a)))
    return verts

def _feasible_fast(tx, ty, rot_deg, outer_cx, outer_cy, outer_side, outer_rot):
    """Fast feasibility: transform all 72 inner vertices, test against outer hex half-planes."""
    transformed = _rigid_transform(_LOCAL_VERTS, tx, ty, rot_deg)
    outer = _outer_edges(outer_cx, outer_cy, outer_side, outer_rot)
    n = len(outer)
    for i in range(n):
        a = np.array(outer[i])
        b = np.array(outer[(i + 1) % n])
        # half-plane: cross(b-a, p-a) >= -eps
        edge = b - a
        cross = (transformed - a)[:, 0] * edge[1] - (transformed - a)[:, 1] * edge[0]
        if np.any(cross < -1e-9):
            return False
    return True

def construct():
    t0 = time.time()
    best = (math.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (score, tx, ty, rot, ocx, ocy, orot)

    # Symmetry: by symmetry we can fix outer centre at origin and outer rotation to 0,
    # and only optimize inner translation (tx,ty) and inner rotation rot.
    # The outer side must be large enough to contain the transformed cluster.
    # For a given (tx,ty,rot), the minimal outer side (with outer centred at origin, rot 0)
    # is the minimal S such that all transformed vertices are inside the hex of side S.
    # For a regular hexagon of side S centred at origin (flat-top, rot 0), the half-planes are:
    #   |x| <= S, and |x/2 + sqrt(3)/2 * y| <= S * sqrt(3)/2  (for the slanted sides)
    # Actually for flat-top hex with vertices at angles 0,60,...: the supporting lines are at distance S*cos(30)=S*sqrt(3)/2 from centre for the 6 sides.
    # The inradius of a regular hexagon of side S is S*sqrt(3)/2.
    # So a point (x,y) is inside iff max over 6 directions of (x*cos(theta)+y*sin(theta)) <= S*sqrt(3)/2,
    # where theta = 30,90,150,210,270,330 (normals to the sides).
    def min_outer_side(tx, ty, rot_deg):
        transformed = _rigid_transform(_LOCAL_VERTS, tx, ty, rot_deg)
        # normals at 30,90,150,210,270,330
        thetas = [math.radians(30 + 60 * k) for k in range(6)]
        maxproj = 0.0
        for th in thetas:
            proj = transformed[:, 0] * math.cos(th) + transformed[:, 1] * math.sin(th)
            m = float(np.max(proj))
            if m > maxproj:
                maxproj = m
        # inradius = S * sqrt(3)/2  =>  S = maxproj / (sqrt(3)/2)
        return maxproj / (math.sqrt(3.0) / 2.0)

    # Optimize (tx, ty, rot) to minimize min_outer_side.
    # Use a coarse grid then Nelder-Mead.
    from scipy.optimize import minimize

    def objective(v):
        tx, ty, rot = v
        return min_outer_side(tx, ty, rot)

    best_val = math.inf
    best_v = None
    # Coarse grid
    for tx in np.linspace(-4, 4, 9):
        for ty in np.linspace(-4, 4, 9):
            for rot in np.linspace(0, 60, 7):
                val = objective([tx, ty, rot])
                if val < best_val:
                    best_val = val
                    best_v = [float(tx), float(ty), float(rot)]
        if time.time() - t0 > DEADLINE:
            break

    # Refine with Nelder-Mead from best grid point
    if best_v is not None and time.time() - t0 < DEADLINE:
        try:
            res = minimize(objective, best_v, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = float(res.fun)
                best_v = [float(x) for x in res.x]
        except Exception:
            pass

    # Also try a few random restarts if time allows
    rng = np.random.default_rng(0)
    for _ in range(8):
        if time.time() - t0 > DEADLINE:
            break
        v0 = [float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)), float(rng.uniform(0, 60))]
        try:
            res = minimize(objective, v0, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = float(res.fun)
                best_v = [float(x) for x in res.x]
        except Exception:
            pass

    if best_v is None:
        # Fallback: trivial valid solution (large outer)
        tx, ty, rot = 0.0, 0.0, 0.0
        side = min_outer_side(tx, ty, rot)
    else:
        tx, ty, rot = best_v
        side = min_outer_side(tx, ty, rot)

    # Build inner list: cluster centres transformed
    inner = []
    for (cx, cy) in _CLUSTER:
        x = cx + tx
        y = cy + ty
        inner.append([x, y, rot])

    # Outer: centred at origin, side = side, rotation 0
    return inner, (0.0, 0.0), side, 0.0
