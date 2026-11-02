import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _find_min_enclosing_hex_side(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length) or None.
    """
    # Normals of the outer hexagon sides
    # Vertices at outer_angle_deg + 60k.
    # Edges between vertices k and k+1.
    # Normal to edge is at outer_angle_deg + 30 + 60k.
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    # Variables: [cx, cy, R] where R is circumradius of outer hex.
    # Minimize R.
    # Constraints for each vertex v and normal n:
    # -R <= (v - c) . n <= R
    # (v - c) . n <= R  =>  -c . n - R <= -v . n
    # -(v - c) . n <= R =>  c . n - R <= v . n
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            # nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except Exception:
        return None

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)
    
    # Generate 11 points on triangular lattice
    # Center at (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 12 neighbors, pick 5 more to make 11?
    # Or just pick the 11 closest points to origin on the lattice.
    
    # Lattice vectors for unit hexagons (side 1)
    # Distance between centers is sqrt(3) for edge-sharing?
    # No, for disjoint hexagons, they can touch.
    # If they touch, distance between centers is sqrt(3) * 1?
    # Let's check.
    # Hexagon side 1. Width (vertex to vertex) 2. Height (flat to flat) sqrt(3).
    # If two hexagons share an edge, the distance between centers is the height of the hexagon?
    # No. The distance from center to edge is sqrt(3)/2.
    # So distance between centers of touching hexagons is 2 * (sqrt(3)/2) = sqrt(3).
    
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2.0, 1.5)
    
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # The inner hexagons can be rotated.
    # Due to symmetry, rotating all inner hexagons by the same angle is equivalent to rotating the outer hexagon by that angle.
    # However, we can also try flipping some? No, regular hexagons are symmetric.
    # So we fix inner angle to 0 and search outer angle from 0 to 30.
    # Or fix outer to 0 and search inner? Same thing.
    
    inner_angle = 0.0
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
    
    # Coarse search for outer angle
    best_coarse_side = float('inf')
    best_coarse_angle = 0.0
    best_coarse_center = (0.0, 0.0)
    
    # Symmetry: 0 to 30 degrees is sufficient
    # Step 0.5 degrees
    for outer_angle_deg in np.arange(0, 30.1, 0.5):
        if time.time() > _DEADLINE:
            break
        res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_angle = outer_angle_deg
                best_coarse_center = (cx, cy)
    
    # Fine search around best coarse angle
    if best_coarse_angle > 0:
        search_range = 0.5
        step = 0.01
        # Ensure we don't go out of bounds if best_coarse_angle is near 0 or 30
        start = max(0, best_coarse_angle - search_range)
        end = min(30, best_coarse_angle + search_range)
        for outer_angle_deg in np.arange(start, end + step, step):
            if time.time() > _DEADLINE:
                break
            res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
    
    if best_inner is None and best_coarse_side < float('inf'):
        best_side = best_coarse_side
        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        best_center = best_coarse_center
        best_outer_angle = best_coarse_angle
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
