"""Structural change: rigid hexagonal cluster of 11 unit hexagons, optimize global rotation
via high-resolution grid search + local refinement to minimize enclosing regular hexagon side."""
import math
import time
import numpy as np

N = 11
DEADLINE = 145.0  # safety margin before 150s kill
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation (degrees)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _support_function(verts, angle_rad):
    """Max projection of polygon vertices onto direction angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    max_proj = -1e18
    for (px, py) in verts:
        proj = px * c + py * s
        if proj > max_proj:
            max_proj = proj
    return max_proj

def _min_enclosing_hex_side(cluster_verts, outer_angle_deg):
    """
    Given a set of polygon vertices (the union of all inner hexagon vertices),
    find the minimum side length of a regular hexagon (centered at origin, rotated by outer_angle_deg)
    that contains all vertices.
    
    For a regular hexagon with side s centered at origin and rotated by angle phi,
    a point (x,y) is inside iff for all 6 facet normals n_k, n_k . (x,y) <= s * cos(pi/6) = s * sqrt(3)/2.
    Wait, let me reconsider. A regular hexagon with side s has inradius r = s * sqrt(3)/2.
    The facet normals are at angles phi + k*pi/3 for k=0..5.
    A point is inside iff |n_k . p| <= r for all k... no, that's for a hexagon centered at origin.
    
    Actually for a regular hexagon centered at origin with inradius r (distance from center to each facet),
    a point p is inside iff n_k . p <= r for all 6 outward normals n_k.
    The minimum enclosing hexagon (centered at origin, fixed rotation) has:
    s = max_k (n_k . p_k_max) / (sqrt(3)/2)  ... but we need to handle both sides.
    
    More carefully: the hexagon is the intersection of 6 half-planes n_k . x <= r and -n_k . x <= r.
    So r >= max_k max_p (n_k . p) and r >= max_k max_p (-n_k . p).
    Thus r_min = max_k max( max_p(n_k.p), max_p(-n_k.p) )
    And s = r_min / (sqrt(3)/2) = 2 * r_min / sqrt(3).
    """
    phi = math.radians(outer_angle_deg)
    sqrt3 = math.sqrt(3.0)
    r_min = 0.0
    for k in range(6):
        alpha = phi + k * math.pi / 3.0
        nx = math.cos(alpha)
        ny = math.sin(alpha)
        max_pos = 0.0
        max_neg = 0.0
        for (px, py) in cluster_verts:
            proj = px * nx + py * ny
            if proj > max_pos:
                max_pos = proj
            if -proj > max_neg:
                max_neg = -proj
        r_candidate = max(max_pos, max_neg)
        if r_candidate > r_min:
            r_min = r_candidate
    s = 2.0 * r_min / sqrt3
    return s

def _build_cluster(angle_deg):
    """
    Build 11 unit hexagons in a rigid hexagonal lattice cluster (center + 6 ring + 4 more? 
    Actually for 11: center + 6 in first ring + 4 in second ring? 
    Standard hexagonal number: 1, 7, 19... 
    For 11, use center + 6 neighbors + 4 more placed to form a compact cluster.
    
    Let's use a simple arrangement: 
    - Center at (0,0)
    - 6 hexagons at distance 2 (adjacent in hex grid) at angles 0,60,120,180,240,300
    - 4 more at distance 4 at angles 30,150,270,330 (or similar)
    
    Actually, let me use a well-known compact packing of 11 hexagons.
    A common arrangement: 
    Row 0 (y=0): 3 hexagons at x = -2, 0, 2
    Row 1 (y=sqrt(3)): 2 hexagons at x = -1, 1  (wait, hex grid spacing)
    
    For unit hexagons (side 1), adjacent centers are distance 2 apart.
    Hex grid: axial coordinates. Let me place them in axial coords and convert.
    
    Axial to cartesian for pointy-top hexagons with side 1:
    x = 2 * (q + r/2) * 1  ... let me use flat-top.
    
    For flat-top hexagons, center spacing:
    x = 3/2 * q
    y = sqrt(3) * (r + q/2)
    Adjacent centers are distance sqrt(3) apart? No.
    
    Let me just use a simple grid. For unit hexagons (side=1), the distance between centers of adjacent hexagons is 2 (they share an edge of length 1, and the center-to-center distance is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)? No.
    
    Inradius of unit hexagon = sqrt(3)/2.
    Distance between centers of two edge-sharing unit hexagons = 2 * inradius = sqrt(3).
    
    So let me place centers on a hex grid with spacing sqrt(3).
    
    Let me use a compact cluster of 11:
    Center at (0,0).
    Ring 1: 6 hexagons at distance sqrt(3), angles 0,60,120,180,240,300.
    Ring 2: We need 4 more. Place them at distance 2*sqrt(3) at angles 30,150,270,330? 
    That might not be compact. 
    
    Alternative: use a "cross" shape or other compact arrangement.
    
    Let me just use a simple linear + offset arrangement that is known to be compact.
    Actually, for the purpose of this structural change, let me use a simple arrangement:
    11 hexagons in a 3-4-3 or similar pattern.
    
    Let me use:
    Row 0 (y=0): x = -2*sqrt(3), 0, 2*sqrt(3)  (3 hexagons)
    Row 1 (y=sqrt(3)/2 * 2 = sqrt(3)): x = -sqrt(3), sqrt(3)  (2 hexagons) -- wait this doesn't tile well.
    
    I'll just use a simple rigid cluster: center + 6 around it + 4 more at the "corners" of the hexagonal lattice.
    """
    sqrt3 = math.sqrt(3.0)
    # Center
    centers = [(0.0, 0.0)]
    # Ring 1: 6 hexagons
    for i in range(6):
        theta = math.radians(60 * i)
        centers.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    # Ring 2: 4 more hexagons at distance 2*sqrt(3), angles 30, 150, 270, 330
    for i in range(4):
        theta = math.radians(30 + 120 * i)
        centers.append((2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)))
    
    # Now rotate the entire cluster by angle_deg and collect all vertices
    rot = math.radians(angle_deg)
    cr = math.cos(rot)
    sr = math.sin(rot)
    
    all_verts = []
    inner = []
    for (cx, cy) in centers:
        # Rotate center
        rcx = cr * cx - sr * cy
        rcy = sr * cx + cr * cy
        inner.append([rcx, rcy, angle_deg])
        # Add vertices of this unit hexagon (flat-top, side=1)
        for j in range(6):
            theta = math.radians(60 * j)
            vx = rcx + math.cos(theta)
            vy = rcy + math.sin(theta)
            all_verts.append((vx, vy))
    
    return inner, all_verts

def construct():
    start = time.time()
    
    best_side = float('inf')
    best_angle = 0.0
    best_inner = None
    best_outer = (0.0, 0.0)
    
    # High-resolution grid search over rotation angle
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees
    n_grid = 1200
    for i in range(n_grid + 1):
        if time.time() - start > DEADLINE - 10:
            break
        angle = 60.0 * i / n_grid
        inner, verts = _build_cluster(angle)
        side = _min_enclosing_hex_side(verts, 0.0)  # outer hexagon at 0 degrees
        if side < best_side:
            best_side = side
            best_angle = angle
            best_inner = inner
            best_outer = (0.0, 0.0)
    
    # Local refinement around best angle
    if time.time() - start < DEADLINE - 5:
        lo = max(0.0, best_angle - 60.0 / n_grid)
        hi = min(60.0, best_angle + 60.0 / n_grid)
        # Golden section search
        gr = (math.sqrt(5) - 1) / 2
        a = lo
        b = hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _min_enclosing_hex_side(_build_cluster(c)[1], 0.0)
        fd = _min_enclosing_hex_side(_build_cluster(d)[1], 0.0)
        for _ in range(50):
            if time.time() - start > DEADLINE - 5:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _min_enclosing_hex_side(_build_cluster(c)[1], 0.0)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _min_enclosing_hex_side(_build_cluster(d)[1], 0.0)
        
        best_angle = (a + b) / 2
        inner, verts = _build_cluster(best_angle)
        side = _min_enclosing_hex_side(verts, 0.0)
        if side < best_side:
            best_side = side
            best_inner = inner
            best_outer = (0.0, 0.0)
    
    if best_inner is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_outer, best_side, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
