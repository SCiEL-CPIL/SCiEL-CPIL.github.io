"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def polygons_overlap_sat(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            # Edge normal
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If there's a gap, no overlap
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def sat_distance(v1, v2):
    """
    Calculate the minimum distance between two convex polygons using SAT.
    Returns 0 if they overlap, otherwise the minimum separation distance.
    """
    min_dist = float('inf')
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9: continue
            normal = normal / norm
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Calculate gap
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0
                
            if gap < min_dist:
                min_dist = gap
                if min_dist < 1e-9:
                    return 0.0
    return min_dist

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of [cx, cy, angle_deg]) and outer hexagon params,
    find the minimum outer side length such that all inner hexagons fit inside.
    """
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    
    max_scale = 0.0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = hex_vertices(cx, cy, ang)
        for vx, vy in verts:
            # Vector from outer center to this vertex
            dx, dy = vx - ox, vy - oy
            required_s = 0.0
            for k in range(6):
                # Direction from center to edge k midpoint
                edge_mid_angle = oa + k * math.pi / 3 + math.pi / 6
                # Inward normal: from edge toward center, opposite to edge_mid_angle
                inward_nx = -math.cos(edge_mid_angle)
                inward_ny = -math.sin(edge_mid_angle)
                dot_val = dx * inward_nx + dy * inward_ny
                # Constraint: dot_val <= s * sqrt(3)/2
                # So s >= dot_val / (sqrt(3)/2) if dot_val > 0
                if dot_val > 0:
                    s_req = dot_val / (math.sqrt(3) / 2)
                    required_s = max(required_s, s_req)
            max_scale = max(max_scale, required_s)
    
    return max_scale

def objective_function(params):
    """
    Objective function for Nelder-Mead.
    params: flat array of [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    Returns: outer_side_length + penalty * sum(overlap_distances)
    """
    # Reshape params
    hexes = params.reshape((N, 3))
    centers = hexes[:, :2]
    angles_deg = hexes[:, 2]
    
    # 1. Calculate minimum outer side length
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0 # We can optimize this too, but for now fix it to 0 for simplicity in the loop, 
                          # or include it in params? 
                          # Let's include outer angle in params? No, keep it simple first. 
                          # Actually, rotating the outer hexagon is equivalent to rotating all inner ones relative to it.
                          # Since inner hexagons can rotate independently, fixing outer at 0 is WLOG 
                          # IF we allow inner rotation. 
                          # Wait, the "score" is the outer side length. 
                          # The outer hexagon can be rotated. 
                          # If we fix outer at 0, we might miss optimal alignments where outer is rotated.
                          # However, since inner hexagons can rotate freely, the relative orientation is what matters.
                          # Let's assume outer is fixed at 0 for the optimization, and we just minimize the size.
                          # Is it better to allow outer rotation? 
                          # If we rotate outer by phi, it's like rotating the coordinate system.
                          # The set of valid configurations is invariant under global rotation.
                          # So fixing outer angle to 0 is without loss of generality for the *size* minimization,
                          # provided we don't constrain inner angles to be multiples of 60.
                          # We allow continuous inner angles.
    
    inner_hexes = [[hexes[i, 0], hexes[i, 1], hexes[i, 2]] for i in range(N)]
    outer_side = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)
    
    # 2. Calculate overlap penalty
    # We need to check all pairs
    penalty = 0.0
    verts_list = [hex_vertices(hexes[i, 0], hexes[i, 1], hexes[i, 2]) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = sat_distance(verts_list[i], verts_list[j])
            if dist < 0: # Overlap
                penalty += -dist * 100.0 # Strong penalty
            elif dist < 0.1: # Soft penalty for near-touching to encourage separation
                penalty += (0.1 - dist) * 10.0
                
    return outer_side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    best_side = float('inf')
    best_params = None
    
    # Pre-compute some constants
    sqrt3 = math.sqrt(3)
    
    # Define initial configurations to try
    inits = []
    
    # 1. Grid: 4 rows x 3 cols
    # Hexagon side 1. Distance between centers in honeycomb:
    # Horizontal: 1.5 (sqrt(3) * side * 0.866? No. 
    # Flat-top hex: width = sqrt(3), height = 2.
    # Pointy-top hex: width = 2, height = sqrt(3).
    # Let's use flat-top (angle 0 is vertex at right? No, standard math angle 0 is (1,0)).
    # My hex_vertices: i=0 -> theta=0 -> (1,0). So vertex at (1,0).
    # This is a "pointy-side" or "flat-top"? 
    # Vertices at 0, 60, 120...
    # Edge from 0 to 60 deg. Midpoint at 30 deg. 
    # The "top" is a flat edge? No, vertex at 90 deg is (0,1). 
    # So it's pointy-top? 
    # Let's check: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866).
    # Top edge is between (0.5, 0.866) and (-0.5, 0.866). Yes, flat top.
    # Distance between centers of adjacent flat-top hexagons:
    # Horizontal neighbor: distance = 1.5 * sqrt(3)? No.
    # Width of hex = 2. 
    # Horizontal spacing for flat-top: 1.5 * width? No.
    # Let's just use a grid that is slightly larger than touching to avoid initial overlaps.
    # Spacing = 2.2 (safe for side 1)
    
    spacing = 2.2
    for row in range(4):
        for col in range(3):
            cx = col * spacing
            cy = row * spacing * sqrt3
            if row % 2 == 1:
                cx += spacing / 2.0
            # Center the grid roughly at 0,0
            # Grid center approx: x=1.5*spacing, y=1.5*spacing*sqrt3
            cx -= 1.5 * spacing
            cy -= 1.5 * spacing * sqrt3
            inits.append([cx, cy, 0.0])
            
    if len(inits) == N:
        inits_flat = np.array(inits).flatten()
        
    # 2. Random initialization
    np.random.seed(42)
    for _ in range(5):
        rand_params = np.random.uniform(-3, 3, size=N*3)
        # Ensure angles are reasonable
        rand_params[2::3] = np.random.uniform(-180, 180, size=N)
        inits.append(rand_params)
        
    # 3. Rotated Grid
    grid_params = np.array(inits[0]).flatten() # The first init was the grid
    # Rotate all centers by 30 degrees
    for i in range(N):
        x = grid_params[i*3]
        y = grid_params[i*3+1]
        a = 30.0
        rad = math.radians(a)
        nx = x * math.cos(rad) - y * math.sin(rad)
        ny = x * math.sin(rad) + y * math.cos(rad)
        grid_params[i*3] = nx
        grid_params[i*3+1] = ny
        grid_params[i*3+2] += 30.0 # Rotate hexagon too
    inits.append(grid_params)
    
    # Optimization Loop
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Nelder-Mead
            # options: maxiter, xatol, fatol
            # We want to be fast but accurate.
            res = minimize(objective_function, init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < best_side:
                # Verify validity of the result
                # Check overlaps strictly
                hexes = res.x.reshape((N, 3))
                verts_list = [hex_vertices(hexes[i, 0], hexes[i, 1], hexes[i, 2]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if polygons_overlap_sat(verts_list[i], verts_list[j], tol=1e-8):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    inner_hexes = [[hexes[i, 0], hexes[i, 1], hexes[i, 2]] for i in range(N)]
                    exact_side = compute_min_outer_side(inner_hexes, (0.0, 0.0), 0.0)
                    if exact_side < best_side:
                        best_side = exact_side
                        best_params = res.x.copy()
                        
        except Exception as e:
            continue

    # Fallback
    if best_params is None:
        # Use the initial grid
        best_params = inits[0]
        hexes = best_params.reshape((N, 3))
        inner_hexes = [[hexes[i, 0], hexes[i, 1], hexes[i, 2]] for i in range(N)]
        best_side = compute_min_outer_side(inner_hexes, (0.0, 0.0), 0.0)

    # Construct return value
    hexes = best_params.reshape((N, 3))
    inner = []
    for i in range(N):
        inner.append([hexes[i, 0], hexes[i, 1], hexes[i, 2]])
    
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
