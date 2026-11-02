import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Precompute base hexagon vertices (flat-top, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300
BASE_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3)/2],
    [-0.5, math.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3)/2],
    [0.5, -math.sqrt(3)/2]
])

def get_hex_vertices(center, angle_deg):
    """
    Returns 6 vertices of a unit hexagon at given center and rotation.
    """
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    verts = (BASE_HEX_VERTS @ rot_matrix.T) + center
    return verts

def polygon_distance(v1, v2):
    """
    Calculate the minimum distance between two convex polygons (hexagons).
    Returns 0 if they intersect or touch.
    Uses a simplified GJK-like approach or edge-to-edge distance check.
    For 6x6 vertices, checking all edge pairs is 36 checks, very fast.
    """
    # Check if any vertex of v1 is inside v2 or vice versa? 
    # Simpler: Check distance between all edges.
    # If polygons intersect, distance is 0.
    
    # Fast check: Bounding box overlap
    min_x1 = np.min(v1[:, 0]); max_x1 = np.max(v1[:, 0])
    min_y1 = np.min(v1[:, 1]); max_y1 = np.max(v1[:, 1])
    min_x2 = np.min(v2[:, 0]); max_x2 = np.max(v2[:, 0])
    min_y2 = np.min(v2[:, 1]); max_y2 = np.max(v2[:, 1])
    
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        # No overlap in bounding boxes, calculate min distance between edges
        min_dist = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                d = seg_seg_dist(p1, p2, p3, p4)
                if d < min_dist:
                    min_dist = d
        return min_dist
    else:
        # Bounding boxes overlap. Check for intersection.
        # If they intersect, distance is 0.
        # We can use a point-in-polygon test for one vertex of v1 against v2.
        if point_in_polygon(v1[0], v2) or point_in_polygon(v2[0], v1):
            return 0.0
        # If not intersecting but bbox overlaps, it's a near-miss, calculate edge dist
        min_dist = float('inf')
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                d = seg_seg_dist(p1, p2, p3, p4)
                if d < min_dist:
                    min_dist = d
        return min_dist

def seg_seg_dist(p1, p2, p3, p4):
    """
    Distance between segment p1-p2 and segment p3-p4.
    """
    # If segments intersect, distance is 0
    if segments_intersect(p1, p2, p3, p4):
        return 0.0
    # Otherwise, min of point-to-segment distances
    d1 = point_seg_dist(p1, p3, p4)
    d2 = point_seg_dist(p2, p3, p4)
    d3 = point_seg_dist(p3, p1, p2)
    d4 = point_seg_dist(p4, p1, p2)
    return min(d1, d2, d3, d4)

def point_seg_dist(p, a, b):
    """
    Distance from point p to segment ab.
    """
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < EPS:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_len_sq
    t = max(0, min(1, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects p3-p4.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def point_in_polygon(p, poly):
    """
    Ray casting algorithm for point in polygon.
    p: (x, y)
    poly: (6, 2) array
    """
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def calculate_outer_side_length(positions, rotations, outer_center, outer_rotation):
    """
    Calculate the minimum side length of the outer hexagon required to contain all inner hexagons.
    """
    # Outer hexagon normals
    outer_theta = math.radians(outer_rotation)
    normals = []
    for i in range(6):
        angle = outer_theta + math.pi/6 + i * math.pi / 3
        normals.append([math.cos(angle), math.sin(angle)])
    normals = np.array(normals)
    
    max_proj = 0.0
    c_out = np.array(outer_center)
    
    for i in range(N):
        verts = get_hex_vertices(positions[i], rotations[i])
        for v in verts:
            diff = v - c_out
            projs = np.abs(normals @ diff)
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    apothem_factor = math.sqrt(3) / 2.0
    return max_proj / apothem_factor

def objective(params):
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf')
        
    # Unpack parameters
    # positions: N*2, rotations: N, outer_center: 2, outer_rotation: 1
    pos_flat = params[:2*N]
    rot_flat = params[2*N:3*N]
    out_center = params[3*N:3*N+2]
    out_rot = params[3*N+2]
    
    positions = pos_flat.reshape(N, 2)
    rotations = rot_flat
    
    # 1. Calculate collision penalty
    penalty = 0.0
    # Precompute vertices for all hexagons to avoid recalculation in inner loop?
    # For N=11, 55 pairs. 55 * (vertex calc + dist calc) is fine.
    verts_list = [get_hex_vertices(positions[i], rotations[i]) for i in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            dist = polygon_distance(verts_list[i], verts_list[j])
            if dist < EPS:
                # Overlapping or touching. Penalty should be large.
                # We want to push them apart. 
                # Since we don't have gradients for the penalty easily, 
                # we use a large value.
                penalty += 1000.0
            elif dist < 0.1:
                # Near collision, strong penalty
                penalty += (0.1 - dist) * 500.0
                
    if penalty > 10.0:
        return 1000.0 + penalty
        
    # 2. Calculate outer side length
    S = calculate_outer_side_length(positions, rotations, out_center, out_rot)
    
    return S + penalty

def construct():
    best_score = float('inf')
    best_params = None
    
    # Strategy: Multiple restarts with diverse initializations.
    # Key: Initialize rotations randomly to break symmetry.
    
    np.random.seed(42)
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Initialization
        # Start with a honeycomb-like structure
        init_pos = []
        init_rot = []
        
        # Center
        init_pos.append([0.0, 0.0])
        init_rot.append(np.random.uniform(-15, 15))
        
        # First ring (6)
        for k in range(6):
            angle = math.pi/3 * k
            # Distance 2.0 is tangent for aligned. With rotation, we might need slightly more or less?
            # Actually, max width is 2.0. So 2.0 is safe minimum.
            init_pos.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
            init_rot.append(np.random.uniform(-30, 30))
            
        # Second ring (4)
        # Place them roughly at distance 4.0
        for k in range(4):
            angle = math.pi/3 * (k + 0.5) + np.random.uniform(-0.2, 0.2)
            rad = 4.0 + np.random.uniform(-0.2, 0.2)
            init_pos.append([rad * math.cos(angle), rad * math.sin(angle)])
            init_rot.append(np.random.uniform(-30, 30))
            
        init_pos = np.array(init_pos)
        init_rot = np.array(init_rot)
        
        # Outer center: Center of mass
        com = np.mean(init_pos, axis=0)
        init_out_rot = np.random.uniform(0, 60)
        
        init_params = np.concatenate([
            init_pos.flatten(),
            init_rot,
            com,
            [init_out_rot]
        ])
        
        # Optimization
        # Use L-BFGS-B for better convergence in higher dimensions
        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 50000})
            if res.fun < best_score and res.fun < 100.0: # Filter out bad initializations
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: Aligned honeycomb
        positions = []
        rotations = []
        positions.append([0.0, 0.0])
        rotations.append(0.0)
        for k in range(6):
            angle = math.pi/3 * k
            positions.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
            rotations.append(0.0)
        for k in range(4):
            angle = math.pi/3 * (k + 0.5)
            positions.append([4.0 * math.cos(angle), 4.0 * math.sin(angle)])
            rotations.append(0.0)
            
        positions = np.array(positions)
        rotations = np.array(rotations)
        com = np.mean(positions, axis=0)
        S = calculate_outer_side_length(positions, rotations, com, 0.0)
        
        inner = []
        for i in range(N):
            inner.append([float(positions[i][0]), float(positions[i][1]), float(rotations[i])])
            
        return inner, [float(com[0]), float(com[1])], float(S), 0.0

    # Extract best solution
    pos_flat = best_params[:2*N]
    rot_flat = best_params[2*N:3*N]
    out_center = best_params[3*N:3*N+2]
    out_rot = best_params[3*N+2]
    
    positions = pos_flat.reshape(N, 2)
    rotations = rot_flat
    
    # Verify disjointness strictly
    verts_list = [get_hex_vertices(positions[i], rotations[i]) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            d = polygon_distance(verts_list[i], verts_list[j])
            if d < 1e-6:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, try to repair by pushing apart? 
        # Or just return fallback?
        # For now, return fallback to ensure validity
        positions = []
        rotations = []
        positions.append([0.0, 0.0])
        rotations.append(0.0)
        for k in range(6):
            angle = math.pi/3 * k
            positions.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
            rotations.append(0.0)
        for k in range(4):
            angle = math.pi/3 * (k + 0.5)
            positions.append([4.0 * math.cos(angle), 4.0 * math.sin(angle)])
            rotations.append(0.0)
        positions = np.array(positions)
        rotations = np.array(rotations)
        com = np.mean(positions, axis=0)
        S = calculate_outer_side_length(positions, rotations, com, 0.0)
        
        inner = []
        for i in range(N):
            inner.append([float(positions[i][0]), float(positions[i][1]), float(rotations[i])])
        return inner, [float(com[0]), float(com[1])], float(S), 0.0

    S = calculate_outer_side_length(positions, rotations, out_center, out_rot)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(rotations[i])])
        
    return inner, [float(out_center[0]), float(out_center[1])], float(S), float(out_rot)

if __name__ == "__main__":
    construct()
