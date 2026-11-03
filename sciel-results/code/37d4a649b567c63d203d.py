"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (flat-top orientation at 0 deg)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _project_range(verts, axis):
    """Min and max projection of polygon vertices onto a unit axis."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            # Normal to edge (2D)
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = _project_range(v1, normal)
            r2min, r2max = _project_range(v2, normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside the hexagon (center cx,cy, side, rotation)."""
    # Transform point to hexagon-local frame
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a
    ca, sa = math.cos(-a), math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes.
    # In local frame with flat-top (angle 0), vertices at angles 0,60,120,180,240,300
    # The 6 edge normals point at angles 30,90,150,210,270,330
    inradius = side * math.sqrt(3.0) / 2.0
    for i in range(6):
        theta = math.pi / 6.0 + i * math.pi / 3.0  # 30, 90, 150, ...
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _inner_fits_outer(inner_verts, outer_verts, tol=1e-9):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_hex(v[0], v[1], 0.0, 0.0, 1.0, 0.0, tol):
            pass  # This check is for the outer, not inner
    # Better: check each inner vertex is inside outer using SAT-style or point-in-poly
    # Use winding number / ray casting for convex polygon
    for v in inner_verts:
        if not _point_in_convex(v[0], v[1], outer_verts, tol):
            return False
    return True

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        # Cross product of edge (i->j) with point (i->p)
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True

def _feasible(outer_side, placements, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if given placements fit inside outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_verts = _outer_vertices(ox, oy, outer_side, outer_angle)
    
    # Precompute inner vertices
    inner_verts_list = []
    for (ix, iy, iangle) in placements:
        inner_verts_list.append(_hex_vertices(ix, iy, iangle, 1.0))
    
    # Check each inner hexagon fits inside outer
    for iv in inner_verts_list:
        if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
            return False
    
    # Check pairwise non-overlap
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                return False
    
    return True

def _grid_search(outer_side, max_iter=5000):
    """Try to find 12 non-overlapping unit hexagons inside a hexagon of given side.
    Uses a grid of candidate positions and angles."""
    # The outer hexagon has inradius = outer_side * sqrt(3)/2
    inradius = outer_side * math.sqrt(3.0) / 2.0
    
    # Candidate positions: grid within the hexagon
    # For a hexagon of side S, the extent in x is from -S to S (flat-top)
    # Actually for pointy-top vs flat-top... let's use flat-top (angle 0)
    # Vertices at angles 0,60,120,180,240,300 from center
    # x extent: -S to S, y extent: -S*sqrt(3)/2 to S*sqrt(3)/2
    
    # Grid of positions
    num_positions = 40
    num_angles = 7  # 0, 30, 60, 90, 120, 150, 180 (but 180=0 for hex)
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    # Generate candidate positions
    candidates = []
    for i in range(num_positions):
        for j in range(num_positions):
            x = -outer_side + 2.0 * outer_side * i / (num_positions - 1)
            y = -outer_side * math.sqrt(3) / 2.0 + outer_side * math.sqrt(3) * j / (num_positions - 1)
            # Check if this position is feasible (center of unit hex inside outer)
            # The unit hex extends 1.0 in all directions (circumradius=1)
            # So we need the center to be at least 1.0 from the outer boundary
            # Quick check: is the center inside the outer hex with margin?
            # For now, just add all and filter later
            for angle in angles:
                candidates.append((x, y, angle))
    
    # Sort candidates by distance from center (prefer central positions)
    candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
    
    # Greedy placement: try to place each hexagon
    placed = []
    placed_verts = []
    
    for (x, y, angle) in candidates:
        if len(placed) >= N:
            break
        # Check if this position is valid
        iv = _hex_vertices(x, y, angle, 1.0)
        
        # Check fits inside outer
        outer_verts = _outer_vertices(0.0, 0.0, outer_side, 0.0)
        fits = True
        for v in iv:
            if not _point_in_convex(v[0], v[1], outer_verts, tol=1e-9):
                fits = False
                break
        if not fits:
            continue
        
        # Check no overlap with already placed
        overlaps = False
        for pv in placed_verts:
            if _sat_overlap(iv, pv, tol=1e-9):
                overlaps = True
                break
        if overlaps:
            continue
        
        placed.append((x, y, angle))
        placed_verts.append(iv)
    
    if len(placed) >= N:
        return placed[:N]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Bisection search on outer side length
    # Lower bound: must fit 12 unit hexagons. Minimum area consideration:
    # Area of outer hex = (3*sqrt(3)/2) * S^2
    # Area of 12 unit hexagons = 12 * (3*sqrt(3)/2) * 1^2 = 6*sqrt(3)
    # So S^2 >= 6, S >= sqrt(6) ≈ 2.449
    # But packing efficiency is less than 100%, so S is larger.
    # Upper bound: place in a line, S ≈ 12 + 2 = 14 or so
    lo = 2.5
    hi = 15.0
    
    best_placements = None
    best_side = hi
    
    # First, find a feasible upper bound
    # Try a few values to bracket
    for test_side in [3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        if time.time() > deadline - 10.0:
            break
        result = _grid_search(test_side)
        if result is not None:
            best_side = test_side
            best_placements = result
            hi = test_side
            break
    
    # If we didn't find any feasible, use a large value
    if best_placements is None:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection to find minimum feasible side
    for _ in range(20):  # ~20 iterations of bisection
        if time.time() > deadline - 15.0:
            break
        mid = (lo + hi) / 2.0
        result = _grid_search(mid)
        if result is not None:
            hi = mid
            best_side = mid
            best_placements = result
        else:
            lo = mid
    
    # Return the best found
    if best_placements is not None:
        return best_placements, (0.0, 0.0), best_side, 0.0
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
