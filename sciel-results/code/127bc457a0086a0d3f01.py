"""Packing 12 unit hexagons in a regular hexagon: rigid honeycomb cluster + 2D rotation search."""
import math
import time

N = 12

# ---- Pre-computed honeycomb cluster of 12 unit hexagons (side=1) ----
# Centers are on the standard hex lattice.  For a regular hexagon of side 1 the
# circumradius is 1 and the distance between adjacent lattice centers is sqrt(3).
# We use the canonical 12-hexagon "flower" arrangement:
#   ring 0: 1 hex at origin
#   ring 1: 6 hexes at distance sqrt(3)
#   ring 2: 5 more hexes (the 6th position of ring 2 is omitted to keep 12)
# Actually the tightest 12 is: 1 center + 6 around + 5 of the next ring.
# Let's use the well-known 12-hexagon packing:
#   Row structure (hex lattice, axial coords):
#   We'll place them explicitly.

def _hex_cluster_centers():
    """Return (x, y) centers for 12 unit hexagons in a tight cluster."""
    s = math.sqrt(3.0)  # distance between adjacent hex centers
    centers = []
    # Ring 0
    centers.append((0.0, 0.0))
    # Ring 1: 6 hexes
    for k in range(6):
        ang = math.pi / 3.0 * k
        centers.append((s * math.cos(ang), s * math.sin(ang)))
    # Ring 2: we need 5 more.  The 6 ring-2 positions are at distance 2*s from origin
    # at angles 0, 60, 120, 180, 240, 300 deg.  Pick 5 that give a compact shape.
    # To keep the cluster as symmetric/compact as possible, omit one.
    # We'll include all 6 ring-2 positions but that's 13.  So we need a different arrangement.
    # Better: use a 3-4-3-2 row layout or the known optimal 12-hexagon packing.
    # Let's use a different tight arrangement: two staggered rows of 4 and two of 2,
    # plus... actually let's just use a 4x3 grid in hex coordinates.
    # Simpler and known to be tight: a 4x3 parallelogram of hexes.
    # Hex lattice: x = (col + row/2) * s, y = row * 1.5
    # 4 columns x 3 rows = 12
    centers = []
    for row in range(3):
        for col in range(4):
            x = (col + row * 0.5) * s
            y = row * 1.5
            centers.append((x, y))
    # Center the cluster at origin
    cx = sum(p[0] for p in centers) / len(centers)
    cy = sum(p[1] for p in centers) / len(centers)
    centers = [(x - cx, y - cy) for x, y in centers]
    return centers

_CLUSTERS = _hex_cluster_centers()

def _hexagon_vertices(cx, cy, angle_rad):
    """Vertices of a regular hexagon with side 1, center (cx,cy), rotation angle_rad."""
    verts = []
    for k in range(6):
        a = angle_rad + math.pi / 3.0 * k
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _outer_side_length(cluster_rotation, outer_rotation):
    """
    Given cluster rotation and outer hexagon rotation (both radians),
    compute the minimum side length of a regular hexagon (centered at origin)
    that contains all 12 unit hexagons.
    
    The outer hexagon with side L, center at origin, rotation phi has support
    function: for each of its 6 outward normals n_i, the constraint is
    max over all points p in the union: dot(p, n_i) <= L * cos(pi/6) = L * sqrt(3)/2.
    So L >= max_i ( max_p dot(p, n_i) ) / (sqrt(3)/2).
    """
    sqrt3 = math.sqrt(3.0)
    best_L = 0.0
    for k in range(6):
        # Outward normal direction of outer hexagon face k
        n_ang = outer_rotation + math.pi / 3.0 * k
        nx = math.cos(n_ang)
        ny = math.sin(n_ang)
        max_dot = -1e18
        for (cx, cy) in _CLUSTERS:
            # For this hexagon center, the max dot product with (nx,ny) over the
            # hexagon (side 1, rotated by cluster_rotation) is:
            # dot(center, n) + support_function_of_unit_hex(n, cluster_rotation)
            # Support function of a unit regular hexagon with rotation theta in direction n:
            # The hexagon has 6 faces.  The support in direction n is the max over
            # its 6 vertices of dot(v, n).  For a unit hexagon centered at origin
            # with rotation theta, vertices are at angles theta + k*pi/3, radius 1.
            # So max dot(v, n) = max_k cos(n_ang - theta - k*pi/3)
            # = cos(min angular distance between n_ang and the 6 vertex angles)
            # Actually for a convex polygon, support in direction n = max over vertices.
            # For a regular hexagon with circumradius 1, the support function in direction
            # alpha is: h(alpha) = cos(delta) where delta is the angle from alpha to the
            # nearest vertex angle, but bounded... Let's just compute via the 6 vertices.
            # More efficient: the support function of a regular hexagon (circumradius 1)
            # in direction phi is: h(phi) = cos(phi mod pi/3 - pi/6 + pi/6)... 
            # Let me just use the formula: for a regular hexagon with circumradius R=1,
            # the support function in direction making angle beta with a vertex is
            # h = cos(beta) for |beta| <= pi/6, and by symmetry it's periodic with pi/3.
            # So h(beta) = cos(beta mod (pi/3) - pi/6 + pi/6)... 
            # Simpler: h(beta) = cos(min_k |beta - k*pi/3|)  ... no that's not right either.
            # Let me just compute: the hexagon vertices are at angles theta + k*pi/3.
            # Support in direction n_ang: max_k cos(n_ang - theta - k*pi/3)
            # = cos(min_k |n_ang - theta - k*pi/3|)  (since cos is decreasing on [0,pi])
            # But we need the actual max, which is cos of the smallest absolute angular difference
            # to any vertex angle, provided that difference is <= pi/2 (which it always is
            # for a hexagon since vertices are pi/3 apart).
            
            # Angular difference from n_ang to nearest vertex angle of this hexagon
            diff = (n_ang - cluster_rotation) % (math.pi / 3.0)
            if diff > math.pi / 6.0:
                diff = math.pi / 3.0 - diff
            # Support of unit hexagon in this direction
            supp = math.cos(diff)
            d = cx * nx + cy * ny + supp
            if d > max_dot:
                max_dot = d
        # Required side length for this face
        L_k = max_dot / (sqrt3 / 2.0)
        if L_k > best_L:
            best_L = L_k
    return best_L

def construct():
    t_start = time.time()
    deadline = t_start + 100.0  # safety margin before 120s limit
    
    # Optimize over cluster_rotation and outer_rotation in [0, pi/3)
    # since both have 6-fold symmetry.
    best_L = 1e18
    best_cr = 0.0
    best_or = 0.0
    
    # Coarse grid
    n_coarse = 60
    for i in range(n_coarse):
        cr = math.pi / 3.0 * i / n_coarse
        for j in range(n_coarse):
            orot = math.pi / 3.0 * j / n_coarse
            L = _outer_side_length(cr, orot)
            if L < best_L:
                best_L = L
                best_cr = cr
                best_or = orot
        if time.time() > deadline - 20.0:
            break
    
    # Local refinement around the best point
    # Use a simple iterative refinement: shrink the search window
    window = math.pi / 3.0 / n_coarse
    for _ in range(4):
        if time.time() > deadline - 10.0:
            break
        # Fine grid in the current window
        n_fine = 30
        for i in range(n_fine):
            cr = best_cr - window + 2 * window * i / (n_fine - 1)
            for j in range(n_fine):
                orot = best_or - window + 2 * window * j / (n_fine - 1)
                L = _outer_side_length(cr, orot)
                if L < best_L:
                    best_L = L
                    best_cr = cr
                    best_or = orot
        window /= 3.0
    
    # Final very fine refinement
    if time.time() < deadline - 5.0:
        window = 1e-4
        n_fine = 20
        for _ in range(2):
            for i in range(n_fine):
                cr = best_cr - window + 2 * window * i / (n_fine - 1)
                for j in range(n_fine):
                    orot = best_or - window + 2 * window * j / (n_fine - 1)
                    L = _outer_side_length(cr, orot)
                    if L < best_L:
                        best_L = L
                        best_cr = cr
                        best_or = orot
            window /= 2.0
    
    # Build the inner hexagons: each at its cluster center, rotated by best_cr
    inner = []
    for (cx, cy) in _CLUSTERS:
        inner.append([cx, cy, math.degrees(best_cr)])
    
    return inner, (0.0, 0.0), best_L, math.degrees(best_or)
