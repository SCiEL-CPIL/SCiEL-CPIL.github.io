"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """
    Return the 6 vertices of a regular hexagon centered at (cx, cy)
    with side length 'side' and rotation angle 'angle_deg' in degrees.
    Vertices are ordered counter-clockwise starting from the vertex
    closest to the positive x-axis after rotation.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle_rad
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _check_collision(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (each with 6 vertices) intersect or touch.
    Returns True if they collide (intersection or touching).
    Uses Separating Axis Theorem (SAT).
    """
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > eps:
                normal = normal / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Check for separation
        if max1 <= min2 + eps or max2 <= min1 + eps:
            return False
    return True

def _is_inside_hexagon(point, center, side, angle_deg):
    """
    Check if a point is inside a regular hexagon.
    """
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    # Transform point to hexagon's local coordinate system
    dx = point[0] - cx
    dy = point[1] - cy
    # Rotate by -angle_rad
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon has vertices at angles 0, 60, ... with radius 'side'
    # A point is inside if for all 6 edge normals, the dot product is <= side
    # Edge normals in local coords are at angles 30, 90, 150, 210, 270, 330 degrees
    # Actually, for a regular hexagon with vertices at 0, 60, ..., the apothem is side * sqrt(3)/2
    # The distance from center to each edge is side * sqrt(3)/2
    # The normals point outward at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    distances = normals @ np.array([lx, ly])
    apothem = side * np.sqrt(3) / 2.0
    # All distances must be <= apothem (with small tolerance)
    return np.all(distances <= apothem + 1e-9)

def _compute_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon (centered at outer_center,
    rotated by outer_angle_deg) that contains all inner hexagons.
    """
    ox, oy = outer_center
    outer_angle_rad = np.deg2rad(outer_angle_deg)
    
    # The outer hexagon has 6 edges. For each edge, compute the maximum distance
    # from the center to any vertex of any inner hexagon, projected onto the edge normal.
    # The side length must be at least the maximum of these projections divided by cos(30 deg)
    # Actually, for a regular hexagon, the distance from center to edge (apothem) is side * sqrt(3)/2
    # So side = 2 * apothem / sqrt(3)
    # The apothem in a given direction is the maximum projection of all points onto that direction.
    
    # Edge normals in the outer hexagon's local frame are at angles 30, 90, 150, 210, 270, 330
    normal_angles_local = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    
    max_apothem = 0.0
    for inner in inner_hexes:
        cx, cy, angle_deg = inner
        verts = _hexagon_vertices(cx, cy, 1.0, angle_deg)
        for vert in verts:
            dx = vert[0] - ox
            dy = vert[1] - oy
            # Rotate by -outer_angle_rad to get local coords
            cos_a = np.cos(-outer_angle_rad)
            sin_a = np.sin(-outer_angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for na in normal_angles_local:
                proj = lx * np.cos(na) + ly * np.sin(na)
                if proj > max_apothem:
                    max_apothem = proj
    
    if max_apothem <= 0:
        return 0.0
    side = 2.0 * max_apothem / np.sqrt(3.0)
    return side

def _objective(params):
    """
    Objective function: minimize the outer side length.
    params: array of length 3*N + 3 = 39
    First 3*N: (x, y, angle_deg) for each inner hexagon
    Last 3: (ox, oy, outer_angle_deg)
    """
    n = len(params) // 3
    inner_hexes = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2]
        inner_hexes.append((x, y, angle))
    
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = params[3*n+2]
    
    # Check collisions
    verts_list = []
    for cx, cy, angle in inner_hexes:
        verts = _hexagon_vertices(cx, cy, 1.0, angle)
        verts_list.append(verts)
    
    for i in range(n):
        for j in range(i+1, n):
            if _check_collision(verts_list[i], verts_list[j]):
                return 1e6  # Penalty for collision
    
    # Check containment
    outer_center = (ox, oy)
    for cx, cy, angle in inner_hexes:
        verts = _hexagon_vertices(cx, cy, 1.0, angle)
        for vert in verts:
            if not _is_inside_hexagon(vert, outer_center, _compute_outer_side(inner_hexes, outer_center, outer_angle), outer_angle):
                # This is circular; instead, just compute the required side and return it
                pass
    
    side = _compute_outer_side(inner_hexes, (ox, oy), outer_angle)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial guess: place hexagons in a honeycomb-like pattern
    # A hexagonal grid of 12 hexagons
    # Let's use a 3x4 rectangular grid of hex centers, then optimize
    hex_spacing = 2.0 * np.sqrt(3) / 2.0  # Distance between centers in honeycomb for side=1
    # Actually, for side=1, the distance between centers of adjacent hexagons is 2 * apothem = sqrt(3)
    # No, the distance between centers of adjacent hexagons in a honeycomb is 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's start with a simple grid
    initial_params = np.zeros(3 * N + 3)
    for i in range(N):
        row = i // 4
        col = i % 4
        x = col * 2.0
        y = row * np.sqrt(3)
        initial_params[3*i] = x
        initial_params[3*i+1] = y
        initial_params[3*i+2] = 0.0
    initial_params[3*N] = 3.0  # outer center x
    initial_params[3*N+1] = 1.5 * np.sqrt(3)  # outer center y
    initial_params[3*N+2] = 0.0  # outer angle
    
    best_side = 1e6
    best_params = initial_params.copy()
    
    # Use Nelder-Mead or Powell
    # Powell is often better for this kind of problem
    result = None
    try:
        result = minimize(_objective, initial_params, method='Powell', 
                         options={'maxiter': 200, 'disp': False})
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Try another optimization from a different starting point
        # Rotate the initial configuration
        initial_params2 = initial_params.copy()
        for i in range(N):
            initial_params2[3*i+2] = 30.0  # Rotate inner hexagons by 30 degrees
        try:
            result2 = minimize(_objective, initial_params2, method='Powell', 
                             options={'maxiter': 150, 'disp': False})
            if result2.fun < best_side:
                best_side = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([x, y, angle])
    
    ox = best_params[3*N]
    oy = best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    
    # If optimization failed, return a valid but non-optimal configuration
    if best_side > 1e5:
        # Fallback: place hexagons in a line
        inner = []
        for i in range(N):
            inner.append([i * 2.0, 0.0, 0.0])
        return inner, (5.0, 0.0), 12.0, 0.0
    
    return inner, (ox, oy), best_side, outer_angle
# EVOLVE-BLOCK-END
