"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import math

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side length and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside the hexagon centered at (cx,cy)."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    angle_rad = -math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # A regular hexagon with side length s has inradius r = s * sqrt(3)/2
    # The hexagon can be described as |x| <= s and |y| <= s*sqrt(3)/2 in the proper orientation
    # For a hexagon with flat top (vertex at 0, s), the conditions are:
    # Actually, let's use the standard form. A unit hexagon centered at origin with one vertex at (1,0):
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with side length s, the distance from center to each side (inradius) is s*sqrt(3)/2.
    # The normal directions to the sides are at angles 30, 90, 150, 210, 270, 330 degrees.
    
    r = side * math.sqrt(3) / 2.0
    for i in range(6):
        theta = math.radians(30 + i * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        dist = lx * nx + ly * ny
        if dist > r + 1e-9:
            return False
    return True

def hexagon_overlap(cx1, cy1, a1, cx2, cy2, a2, eps=1e-7):
    """Check if two unit hexagons overlap using SAT (Separating Axis Theorem)."""
    # Get vertices
    v1 = hex_vertices(cx1, cy1, 1.0, a1)
    v2 = hex_vertices(cx2, cy2, 1.0, a2)
    
    # Get edges for both hexagons
    def get_edges(verts):
        edges = []
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i+1) % 6]
            dx = x2 - x1
            dy = y2 - y1
            # Normal to the edge
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            nx = -dy / length
            ny = dx / length
            edges.append((nx, ny))
        return edges
    
    edges1 = get_edges(v1)
    edges2 = get_edges(v2)
    
    for nx, ny in edges1 + edges2:
        # Project all vertices onto the normal
        proj1 = [vx * nx + vy * ny for vx, vy in v1]
        proj2 = [vx * nx + vy * ny for vx, vy in v2]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        if max1 < min2 - eps or max2 < min1 - eps:
            return False  # Found a separating axis
    
    return True

def compute_outer_side_length(inner_positions, outer_center, outer_rotation):
    """Compute the minimum side length of an outer hexagon (with given center and rotation) 
    that contains all inner hexagons."""
    cx, cy = outer_center
    rot = outer_rotation
    
    min_side = 0.0
    
    for (ix, iy, ia) in inner_positions:
        # Check each vertex of the inner hexagon
        v_inner = hex_vertices(ix, iy, 1.0, ia)
        
        for (vx, vy) in v_inner:
            # Transform to outer hexagon's local coordinate system
            dx = vx - cx
            dy = vy - cy
            angle_rad = -math.radians(rot)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # For a hexagon with side length S centered at origin with vertex at (S, 0):
            # The hexagon is defined by |x| <= S and the other constraints.
            # Actually, the constraint for a point (lx, ly) to be inside a hexagon of side S
            # with vertex at (S,0) is:
            # The hexagon vertices are at angles 0, 60, 120, 180, 240, 300.
            # The edges are at angles 30, 90, 150, 210, 270, 330 from the center.
            # The inradius is S*sqrt(3)/2.
            # The condition is: for each normal direction at angle (30 + 60*i) degrees,
            # the projection of (lx, ly) onto that normal must be <= S*sqrt(3)/2.
            
            r = math.sqrt(3) / 2.0
            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                dist = lx * nx + ly * ny
                required_S = dist / r
                if required_S > min_side:
                    min_side = required_S
    
    return min_side

def is_valid_configuration(inner_positions, outer_center, outer_side, outer_rotation):
    """Check if the configuration is valid: no overlapping inner hexagons, all inside outer."""
    # Check inner hexagons don't overlap
    for i in range(N):
        for j in range(i+1, N):
            ix1, iy1, ia1 = inner_positions[i]
            ix2, iy2, ia2 = inner_positions[j]
            if hexagon_overlap(ix1, iy1, ia1, ix2, iy2, ia2):
                return False
    
    # Check all inner hexagons are inside outer
    if compute_outer_side_length(inner_positions, outer_center, outer_rotation) > outer_side + 1e-7:
        return False
    
    return True

def objective(params):
    """Objective function for optimization. Returns outer side length + penalty for violations."""
    # Unpack parameters
    # For each inner hexagon: x, y, angle (3 params each)
    # Outer: cx, cy, rotation (3 params)
    # Total: N*3 + 3 = 39 params
    
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append([x, y, a])
    
    outer_cx = params[N*3]
    outer_cy = params[N*3+1]
    outer_rot = params[N*3+2]
    
    outer_center = (outer_cx, outer_cy)
    
    # Compute required outer side length
    required_side = compute_outer_side_length(inner, outer_center, outer_rot)
    
    # Penalty for overlapping inner hexagons
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            if hexagon_overlap(ix1, iy1, ia1, ix2, iy2, ia2):
                # Estimate overlap amount - use distance between centers as proxy
                dist = math.sqrt((ix1-ix2)**2 + (iy1-iy2)**2)
                # Two unit hexagons need at least some distance
                penalty += max(0, 2.5 - dist) ** 2
    
    return required_side + 100.0 * penalty

def construct():
    # Initial guess: arrange hexagons in a grid-like pattern
    # Use a honeycomb-like arrangement as starting point
    np.random.seed(42)
    
    # Start with a reasonable initial configuration
    # Place hexagons in a roughly circular/hexagonal arrangement
    inner = []
    
    # Try a symmetric arrangement
    # Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # Ring of hexagons around center
    # For a hexagonal packing, neighbors are at distance ~2*sqrt(3) in certain directions
    # Actually, for unit hexagons, the minimum center-to-center distance for non-overlapping
    # is about 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.46 for flat-to-flat, but it depends on orientation.
    
    # Let's use a simpler approach: start with a grid and optimize
    # Hexagonal grid spacing
    dx = 2.0 * math.sqrt(3)  # ~3.46
    dy = 2.0 * math.sqrt(3)  # ~3.46
    
    # Place in a hexagonal pattern
    positions = []
    # Center
    positions.append((0, 0))
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = i * math.pi / 3
        x = dx * math.cos(angle)
        y = dx * math.sin(angle)
        positions.append((x, y))
    # Ring 2: add more to get to 12
    for i in range(6):
        angle = i * math.pi / 3 + math.pi / 6
        x = 2 * dx * math.cos(angle)
        y = 2 * dx * math.sin(angle)
        positions.append((x, y))
    
    # Take first 12
    positions = positions[:12]
    
    for (x, y) in positions:
        inner.append([x, y, 0.0])
    
    # Initial outer
    outer_center = (0.0, 0.0)
    outer_rot = 0.0
    
    # Compute initial side length
    initial_side = compute_outer_side_length(inner, outer_center, outer_rot)
    
    # Parameters for optimization
    x0 = []
    for (x, y, a) in inner:
        x0.extend([x, y, a])
    x0.extend([outer_center[0], outer_center[1], outer_rot])
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    # Extract results
    best_params = result.x
    best_inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner.append([x, y, a])
    
    best_outer_cx = best_params[N*3]
    best_outer_cy = best_params[N*3+1]
    best_outer_rot = best_params[N*3+2]
    
    best_side = compute_outer_side_length(best_inner, (best_outer_cx, best_outer_cy), best_outer_rot)
    
    # Try a few more random restarts if time permits
    best_score = best_side
    
    for restart in range(3):
        np.random.seed(42 + restart * 100)
        
        # Random initial configuration
        inner_rand = []
        for i in range(N):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            a = np.random.uniform(0, 180)
            inner_rand.append([x, y, a])
        
        outer_cx_rand = np.random.uniform(-5, 5)
        outer_cy_rand = np.random.uniform(-5, 5)
        outer_rot_rand = np.random.uniform(0, 180)
        
        x0_rand = []
        for (x, y, a) in inner_rand:
            x0_rand.extend([x, y, a])
        x0_rand.extend([outer_cx_rand, outer_cy_rand, outer_rot_rand])
        
        try:
            result_rand = minimize(objective, x0_rand, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            best_params_rand = result_rand.x
            best_inner_rand = []
            for i in range(N):
                x = best_params_rand[i*3]
                y = best_params_rand[i*3+1]
                a = best_params_rand[i*3+2]
                best_inner_rand.append([x, y, a])
            
            best_outer_cx_rand = best_params_rand[N*3]
            best_outer_cy_rand = best_params_rand[N*3+1]
            best_outer_rot_rand = best_params_rand[N*3+2]
            
            rand_side = compute_outer_side_length(best_inner_rand, 
                                                  (best_outer_cx_rand, best_outer_cy_rand), 
                                                  best_outer_rot_rand)
            
            if rand_side < best_score:
                best_score = rand_side
                best_inner = best_inner_rand
                best_outer_cx = best_outer_cx_rand
                best_outer_cy = best_outer_cy_rand
                best_outer_rot = best_outer_rot_rand
        except:
            continue
    
    return best_inner, (best_outer_cx, best_outer_cy), best_score, best_outer_rot

# EVOLVE-BLOCK-END
