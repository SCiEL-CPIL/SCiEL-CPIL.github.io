"""Improved program: structured hexagonal lattice packing of unit hexagons in a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside the hexagon using half-plane tests."""
    angle = math.radians(angle_deg)
    # The 6 inward normals point to the midpoints of edges
    # For a regular hexagon, the apothem is side * sqrt(3)/2
    apothem = side * math.sqrt(3.0) / 2.0
    for i in range(6):
        a = angle + i * math.pi / 3.0 + math.pi / 6.0  # normal to edge i
        dx = px - cx
        dy = py - cy
        dist = dx * math.cos(a) + dy * math.sin(a)
        if dist > apothem + 1e-12:
            return False
    return True

def hexagon_overlap_check(v1, v2, eps=1e-10):
    """
    Check if two hexagons (arrays of 6 vertices each) are disjoint.
    Uses separating axis theorem: check if any edge of either hexagon
    provides a separating axis.
    Returns True if they overlap (not disjoint), False if disjoint.
    """
    # For each edge of hex1 and hex2, check separation
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal (perpendicular)
            nx = -ey
            ny = ex
            # Project both hexagons onto this normal
            proj1 = np.array([v[0] * nx + v[1] * ny for v in v1])
            proj2 = np.array([v[0] * nx + v[1] * ny for v in v2])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If there's a gap, they're separated
            if max1 < min2 - eps or max2 < min1 - eps:
                return False  # Disjoint
    # Also need to check if one is inside the other (shouldn't happen for convex disjoint)
    # For convex polygons, if no separating axis found, they overlap
    return True  # Overlapping

def are_disjoint_hexagons(v1, v2, eps=1e-10):
    """Return True if two hexagons are disjoint (no overlap)."""
    # SAT: if any axis separates them, they're disjoint
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx = -ey
            ny = ex
            proj1 = np.array([v[0] * nx + v[1] * ny for v in v1])
            proj2 = np.array([v[0] * nx + v[1] * ny for v in v2])
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True  # Disjoint
    return False  # Overlapping

def all_disjoint(vertices_list, eps=1e-10):
    """Check all pairs for disjointness."""
    n = len(vertices_list)
    for i in range(n):
        for j in range(i + 1, n):
            if not are_disjoint_hexagons(vertices_list[i], vertices_list[j], eps):
                return False
    return True

def all_inside_outer(inner_verts, outer_verts, eps=1e-10):
    """Check all inner hexagons are inside the outer hexagon."""
    for iv in inner_verts:
        for p in iv:
            if not point_in_hexagon(p[0], p[1], 
                                     np.mean(outer_verts[:, 0]), 
                                     np.mean(outer_verts[:, 1]),
                                     0, 0):  # This is a rough check
                pass
    # Better: use point-in-convex-polygon
    for iv in inner_verts:
        for p in iv:
            if not _point_in_convex_poly(p, outer_verts, eps):
                return False
    return True

def _point_in_convex_poly(p, poly, eps=1e-10):
    """Check if point p is inside convex polygon poly (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross > eps:
            if sign < 0:
                return False
            sign = 1
        elif cross < -eps:
            if sign > 0:
                return False
            sign = -1
    return True

def outer_side_length_for_packing(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Given inner hexagon centers and angles, find the minimum outer hexagon side length
    (with the given outer center and angle) that contains all inner hexagons.
    """
    # The outer hexagon must contain all vertices of all inner hexagons.
    # For a fixed outer hexagon orientation and center, the required side length
    # is determined by the maximum distance from center to any inner vertex,
    # projected onto the 6 apothem directions.
    
    cx, cy = outer_center
    angle = math.radians(outer_angle)
    
    max_proj = 0.0
    for pos, ang in zip(inner_positions, inner_angles):
        verts = hexagon_vertices(pos[0], pos[1], 1.0, ang)
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            for i in range(6):
                a = angle + i * math.pi / 3.0 + math.pi / 6.0
                proj = dx * math.cos(a) + dy * math.sin(a)
                if proj > max_proj:
                    max_proj = proj
    
    # The apothem of the outer hexagon must be >= max_proj
    # side = apothem / (sqrt(3)/2) = 2*apothem/sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def objective(params, inner_angles, outer_center, outer_angle):
    """Objective: outer side length given inner positions."""
    inner_positions = np.array(params).reshape(-1, 2)
    # Check disjointness
    verts_list = []
    for pos, ang in zip(inner_positions, inner_angles):
        verts_list.append(hexagon_vertices(pos[0], pos[1], 1.0, ang))
    
    # Check if all inner hexagons are disjoint
    if not all_disjoint(verts_list, eps=1e-9):
        # Penalize overlap
        overlap_penalty = 0.0
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                # Compute minimum penetration depth (approximate)
                # Use distance between centers as a rough measure
                d = np.linalg.norm(inner_positions[i] - inner_positions[j])
                # Two unit hexagons need center distance >= 2*sqrt(3) ~ 3.464 if aligned
                # But with rotation it's complex; use a penalty
                if d < 3.4:
                    overlap_penalty += (3.4 - d) ** 2
        side = outer_side_length_for_packing(inner_positions, inner_angles, outer_center, outer_angle)
        return side + 100.0 * overlap_penalty
    
    # Check if all inside a reasonable outer hexagon
    side = outer_side_length_for_packing(inner_positions, inner_angles, outer_center, outer_angle)
    
    # Verify all inside
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], side, outer_angle)
    if not all_inside_outer(verts_list, outer_verts, eps=1e-8):
        return side + 1000.0
    
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with hexagonal lattice
    # For N hexagons in a hexagonal arrangement:
    # The optimal packing for hexagons in a hexagon uses a hexagonal lattice
    # with spacing sqrt(3) (distance between centers of adjacent unit hexagons)
    
    # Hexagonal lattice positions
    positions = []
    angles = []
    
    # Generate hexagonal lattice points
    # For a hexagonal number H(k) = 3k(k-1)+1
    # N=11 is not a hexagonal number, so we use a subset
    
    # Hexagonal lattice: positions at (i*a + j*a/2, j*a*sqrt(3)/2)
    # where a = sqrt(3) is the center-to-center distance for touching unit hexagons
    a = math.sqrt(3.0)
    
    # Generate lattice points in a hexagonal shape
    lattice_points = []
    for j in range(-2, 3):  # rows
        for i in range(-2, 3):  # columns
            x = i * a + j * a / 2.0
            y = j * a * math.sqrt(3.0) / 2.0
            lattice_points.append((x, y))
    
    # Sort by distance from center to get the N closest
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take N points
    init_positions = lattice_points[:N]
    init_angles = [0.0] * N
    
    # Convert to flat params
    init_params = np.array(init_positions).flatten()
    
    best_side = float('inf')
    best_params = init_params.copy()
    best_angles = init_angles.copy()
    
    # Try a few outer orientations
    outer_angles_to_try = [0.0, 30.0]
    outer_center = (0.0, 0.0)
    
    for outer_angle in outer_angles_to_try:
        if time.time() > deadline:
            break
        
        # Evaluate initial
        side = objective(init_params, init_angles, outer_center, outer_angle)
        if side < best_side:
            best_side = side
            best_params = init_params.copy()
            best_angles = init_angles.copy()
        
        # Optimize inner positions
        result = minimize(
            lambda p: objective(p, init_angles, outer_center, outer_angle),
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        
        if result.success or result.fun < best_side:
            side = result.fun
            if side < best_side:
                best_side = side
                best_params = result.x.copy()
        
        if time.time() > deadline:
            break
    
    # Extract final positions
    final_positions = np.array(best_params).reshape(-1, 2)
    
    # Compute final outer side length
    final_side = outer_side_length_for_packing(final_positions, best_angles, outer_center, 0.0)
    
    # Build inner list
    inner = []
    for pos, ang in zip(final_positions, best_angles):
        inner.append([pos[0], pos[1], ang])
    
    return inner, outer_center, final_side, 0.0

# EVOLVE-BLOCK-END
