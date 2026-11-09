"""Packing unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, s, rot_deg):
    """Vertices of regular hexagon (flat-top when rot=0) centered at (cx,cy), side s, rotation deg."""
    rot = np.radians(rot_deg)
    # flat-top: vertices at angles 0,60,...,300
    ang = np.deg2rad(np.arange(0, 360, 60)) + rot
    return np.column_stack([cx + s * np.cos(ang), cy + s * np.sin(ang)])

def _point_in_hex(px, py, cx, cy, s, rot_deg):
    """Check if point (px,py) is inside hexagon centered (cx,cy) side s rotation rot_deg."""
    rot = np.radians(rot_deg)
    dx, dy = px - cx, py - cy
    # rotate point into hexagon frame
    ca, sa = np.cos(-rot), np.sin(-rot)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # regular hexagon (pointy-top in local coords with vertices at 30,90,...) 
    # Actually for our flat-top definition, local axes: use apothem tests
    a = s * np.sqrt(3) / 2.0  # apothem
    # For a regular hexagon with vertices at angles 0,60,..., the interior test:
    # |x| <= s and |x| + 2*|y|/sqrt(3) <= 2*s  (for pointy-top)
    # Let's just use half-plane test against edges
    verts = np.column_stack([s * np.cos(np.deg2rad(np.arange(0,360,60))),
                             s * np.sin(np.deg2rad(np.arange(0,360,60)))])
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        ex, ey = x2 - x1, y2 - y1
        # normal pointing inward (toward center): rotate edge by +90 and orient
        nx, ny = -ey, ex
        # orient toward center
        if nx * (0 - x1) + ny * (0 - y1) < 0:
            nx, ny = -nx, -ny
        # point must be on inside
        if nx * lx + ny * ly > nx * x1 + ny * y1 + 1e-9:
            return False
    return True

def _hex_hex_collision(cx1, cy1, r1, cx2, cy2, r2):
    """Two unit regular hexagons (rotations don't matter for collision if both unit & aligned).
    For same-orientation unit hexagons, they collide iff center distance < 2*apothem*? 
    Actually two congruent regular hexagons (same orientation) are disjoint iff 
    center distance >= 2 * (inradius) is NOT sufficient; the minimum center distance for
    non-overlap of two unit regular hexagons (same orientation) is 2 * sqrt(3)/2 * 2 = 2*sqrt(3)? 
    No: two touching unit hexagons (sharing an edge) have center distance = 2 * apothem = 2*(sqrt(3)/2) = sqrt(3).
    So collision iff dist < sqrt(3) - eps (for same orientation).
    For different rotations it's more complex; we keep same orientation to simplify.
    """
    dx, dy = cx1 - cx2, cy1 - cy2
    d2 = dx * dx + dy * dy
    min_d2 = (np.sqrt(3.0)) ** 2  # = 3.0
    return d2 < min_d2 - 1e-7

def _outer_contains_inner(cx, cy, S, rot, inner_pts):
    """Check all inner hexagon vertices inside outer hexagon (center cx,cy side S rot)."""
    for (ix, iy, irot) in inner_pts:
        verts = _hex_vertices(ix, iy, 1.0, irot)
        for (vx, vy) in verts:
            if not _point_in_hex(vx, vy, cx, cy, S, rot):
                return False
    return True

def _objective(params, inner_fixed):
    """params = [cx, cy, S, rot_deg]. inner_fixed: list of (x,y,rot)."""
    cx, cy, S, rot = params
    if S < 1.0:
        return 1e6
    # collision check
    n = len(inner_fixed)
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_hex_collision(inner_fixed[i][0], inner_fixed[i][1],
                                  inner_fixed[j][0], inner_fixed[j][1]):
                return 1e6
    # containment
    if not _outer_contains_inner(cx, cy, S, rot, inner_fixed):
        return 1e6
    return S

# EVOLVE-BLOCK-START
def construct():
    # Build a hexagonal lattice of N unit hexagons.
    # Hex lattice spacing: for same-orientation unit hexagons, neighbors at distance sqrt(3).
    # Use axial coordinates. Generate N points in a hexagonal cluster.
    # Flat-top hex lattice: basis vectors
    #   e1 = (sqrt(3), 0)          (horizontal neighbor)
    #   e2 = (sqrt(3)/2, 1.5)      (upper-right neighbor)  since 60 deg
    # Actually for flat-top regular hexagons tiling, the lattice is:
    #   horizontal spacing = sqrt(3) * ? Let's use the standard: 
    #   center-to-center for edge-sharing = 2 * apothem = sqrt(3).
    # Lattice vectors at 60 deg: v1=(sqrt(3),0), v2=(sqrt(3)/2, 3/2)
    sqrt3 = np.sqrt(3.0)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2.0, 1.5])

    # Generate N lattice points in hexagonal shape centered at origin.
    # Use axial coords (q,r) -> position = q*v1 + r*v2
    # Choose a set of (q,r) that forms a hexagon of "radius" R in axial coords, count them, take N.
    # Generate all (q,r) with |q|+|r|+|q+r| <= 2*R (hex radius R in cube coords)
    pts = []
    R = 0
    while len(pts) < N:
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                s = -q - r
                if abs(q) + abs(r) + abs(s) <= 2 * R:
                    pos = q * v1 + r * v2
                    pts.append((float(pos[0]), float(pos[1]), 0.0))
        R += 1
    # take first N (they fill from center outward due to loop order? not strictly; sort by radius)
    pts = sorted(pts, key=lambda p: p[0] ** 2 + p[1] ** 2)
    inner = pts[:N]

    # Initial outer guess
    # Bounding radius of inner cluster
    max_r = max(np.hypot(p[0], p[1]) for p in inner)
    S0 = max_r + 2.0
    rot0 = 0.0
    cx0, cy0 = 0.0, 0.0

    # Optimize outer hexagon (center, side, rotation) to minimize S
    x0 = np.array([cx0, cy0, S0, rot0])
    res = minimize(_objective, x0, args=(inner,), method='Nelder-Mead',
                   options={'maxiter': 20000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})

    if res.success or res.fun < 1e5:
        cx, cy, S, rot = res.x
        if S < 1e5:
            # sanity: verify
            if _outer_contains_inner(cx, cy, S, rot, inner) and not any(
                _hex_hex_collision(inner[i][0], inner[i][1], inner[j][0], inner[j][1])
                for i in range(len(inner)) for j in range(i + 1, len(inner))
            ):
                return inner, (float(cx), float(cy)), float(S), float(rot)

    # Fallback: generous outer
    S = max_r + 3.0
    return inner, (0.0, 0.0), float(S), 0.0
# EVOLVE-BLOCK-END
