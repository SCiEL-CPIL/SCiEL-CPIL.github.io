import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def polygon_area(verts):
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return abs(area) / 2.0

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with side length s, the distance from center to each edge is s*sqrt(3)/2
    d = side * np.sqrt(3) / 2
    # The 6 edges are perpendicular to directions at 0, 60, 120, 180, 240, 300
    for i in range(6):
        theta = i * np.pi / 3
        ex = np.cos(theta)
        ey = np.sin(theta)
        # Distance from point to the edge line
        dist = lx * ex + ly * ey
        if dist > d:
            return False
    return True

def hexagon_collision(v1, v2):
    """Check if two hexagons (each defined by 6 vertices) are disjoint (no overlap).
    Returns True if they collide (overlap)."""
    # Use separating axis theorem for convex polygons
    axes = []
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            edge = verts[j] - verts[i]
            # Normal to edge
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-10:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        # Project both polygons onto axis
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            return False  # Separated
    return True  # Overlap

def outer_hexagon_side(points, angle_deg=0.0):
    """Find minimum side length of a regular hexagon (with given rotation) that contains all points."""
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    
    # Transform points to hexagon's local frame
    tx = points[:, 0] * cos_a - points[:, 1] * sin_a
    ty = points[:, 0] * sin_a + points[:, 1] * cos_a
    
    # For a regular hexagon with flat-top orientation (vertices at 0,60,120,180,240,300),
    # the constraint is that for each of the 6 edge normals, the projection is bounded.
    # Edge normals are at angles 0, 60, 120, 180, 240, 300
    # The distance from center to each edge is s*sqrt(3)/2
    # So we need: max projection along each normal - min projection along opposite normal <= 2 * s*sqrt(3)/2 = s*sqrt(3)
    
    max_proj = 0
    for i in range(3):
        theta = i * np.pi / 3
        ex = np.cos(theta)
        ey = np.sin(theta)
        proj = tx * ex + ty * ey
        range_i = np.max(proj) - np.min(proj)
        # range_i <= s * sqrt(3)
        s_i = range_i / np.sqrt(3)
        if s_i > max_proj:
            max_proj = s_i
    return max_proj

def objective(params):
    """Objective: minimize outer hexagon side length.
    params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, cx, cy, outer_angle]
    """
    n_params = 3 * N + 3
    if len(params) != n_params:
        return 1e10
    
    positions = []
    angles = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        positions.append([x, y])
        angles.append(a)
    
    cx = params[3*N]
    cy = params[3*N+1]
    outer_angle = params[3*N+2]
    
    # Compute all vertices
    all_verts = []
    for i in range(N):
        verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        all_verts.append(verts)
    
    # Check collisions - if any collision, return large penalty
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_collision(all_verts[i], all_verts[j]):
                # Penalty based on overlap depth approximation
                return 1e6
    
    # Collect all vertices to compute enclosing hexagon
    all_points = np.vstack(all_verts)
    
    # Compute minimum enclosing hexagon side
    side = outer_hexagon_side(all_points, outer_angle)
    
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: hexagonal lattice arrangement
    # 11 hexagons: center + ring of 6 + 4 more
    # Let's use a compact arrangement
    # Hexagonal lattice spacing: distance between centers of adjacent hexagons = 2*sqrt(3)/2 * ... 
    # Actually for flat-top hexagons with side 1, the center-to-center distance for adjacent hexagons is sqrt(3)
    
    # Let's try a 3-4-4 arrangement or similar compact shape
    # Standard hexagonal packing: rows offset
    # For 11 hexagons, try:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # Or: center + 6 around + 4 more
    
    # Let's use a hexagonal lattice with spacing sqrt(3)
    spacing = np.sqrt(3)
    
    # Initial positions in hexagonal lattice
    init_positions = []
    # Center
    init_positions.append([0.0, 0.0])
    # Ring of 6
    for i in range(6):
        angle = i * np.pi / 3
        init_positions.append([spacing * np.cos(angle), spacing * np.sin(angle)])
    # 4 more: extend the hexagon
    # Place them at positions that extend the cluster
    for i in range(4):
        angle = i * np.pi / 3
        init_positions.append([2 * spacing * np.cos(angle), 2 * spacing * np.sin(angle)])
    
    # Build initial params
    params = []
    for pos in init_positions:
        params.extend([pos[0], pos[1], 0.0])  # angle 0
    params.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    params = np.array(params)
    
    best_params = params.copy()
    best_score = objective(params)
    
    # Try multiple random restarts
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            p0 = params.copy()
        else:
            # Perturb initial positions
            p0 = params.copy()
            for i in range(N):
                p0[i*3] += np.random.uniform(-0.5, 0.5)
                p0[i*3+1] += np.random.uniform(-0.5, 0.5)
                p0[i*3+2] += np.random.uniform(-10, 10)
            p0[3*N] = np.random.uniform(-1, 1)
            p0[3*N+1] = np.random.uniform(-1, 1)
            p0[3*N+2] = np.random.uniform(0, 60)
        
        try:
            result = minimize(objective, p0, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best solution
    positions = []
    angles = []
    for i in range(N):
        positions.append([best_params[i*3], best_params[i*3+1]])
        angles.append(best_params[i*3+2])
    
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    # Verify no collisions in final result
    all_verts = []
    for i in range(N):
        verts = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        all_verts.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_collision(all_verts[i], all_verts[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (cx, cy), best_score, outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
