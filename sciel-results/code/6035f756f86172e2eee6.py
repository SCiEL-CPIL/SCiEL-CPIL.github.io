"""Packing 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 120.0  # 120s limit, use 110s for safety

def hex_vertices(cx, cy, angle_deg):
    """Get vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    # Unit hexagon: circumradius = 1, vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    angles = base_angles + a
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def hex_inradius():
    return np.sqrt(3.0) / 2.0

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside a regular hexagon with given center, side, rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(-angle_deg)
    lx = dx * np.cos(a) - dy * np.sin(a)
    ly = dx * np.sin(a) + dy * np.cos(a)
    
    # For a regular hexagon with side length s, in the local frame:
    # Vertices at (s*cos(k*60), s*sin(k*60)) for k=0..5
    # The hexagon is the intersection of 6 half-planes
    # For side s, the inradius is s*sqrt(3)/2
    # A point (lx, ly) is inside if for each face normal, the projection <= inradius
    
    inr = side * np.sqrt(3.0) / 2.0
    # Face normals point in directions 30, 90, 150, 210, 270, 330 degrees
    face_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    for fa in face_angles:
        proj = lx * np.cos(fa) + ly * np.sin(fa)
        if proj > inr + 1e-9:
            return False
    return True

def hex_hex_distance(v1, v2):
    """
    Minimum distance between two unit hexagons.
    v1, v2: (cx1, cy1, angle1, cx2, cy2, angle2)
    Returns negative if overlapping (penetration depth), positive if separated.
    We use the minimum distance between edges.
    """
    cx1, cy1, a1 = v1[0], v1[1], v1[2]
    cx2, cy2, a2 = v2[0], v2[1], v2[2]
    
    verts1 = hex_vertices(cx1, cy1, a1)
    verts2 = hex_vertices(cx2, cy2, a2)
    
    # Check all edge pairs
    min_dist = np.inf
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        for j in range(6):
            p3 = verts2[j]
            p4 = verts2[(j + 1) % 6]
            d = seg_seg_dist(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    
    # Also check if one is inside the other (shouldn't happen for disjoint, but for safety)
    # Check center of one against the other
    if point_in_hexagon(cx1, cy1, cx2, cy2, 1.0, a2):
        return -min_dist if min_dist > 0 else -0.1
    if point_in_hexagon(cx2, cy2, cx1, cy1, 1.0, a1):
        return -min_dist if min_dist > 0 else -0.1
    
    return min_dist

def seg_seg_dist(p1, p2, p3, p4):
    """Distance between two line segments."""
    # Use vectorized approach
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12:
        # p1 is a point
        if e < 1e-12:
            return np.linalg.norm(p1 - p3)
        t = f / e
        t = max(0, min(1, t))
        closest = p3 + t * d2
        return np.linalg.norm(p1 - closest)
    
    if e < 1e-12:
        # p4 is a point
        t = np.dot(d1, r) / a
        t = max(0, min(1, t))
        closest = p1 + t * d1
        return np.linalg.norm(p4 - closest)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    s = 0.0
    t = 0.0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (f + b * s) / e
        t = max(0, min(1, t))
    else:
        # Parallel
        s = 0.0
        t = f / e
        t = max(0, min(1, t))
    
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return np.linalg.norm(closest1 - closest2)

def compute_outer_side(positions, angles, outer_center, outer_angle):
    """
    Given inner hexagon centers and angles, compute the minimum outer hexagon side
    that contains all of them.
    
    positions: (N, 2) array of centers
    angles: (N,) array of angles in degrees
    outer_center: (2,) 
    outer_angle: angle of outer hexagon in degrees
    
    Returns: side length of outer hexagon
    """
    N = len(positions)
    outer_inr = np.sqrt(3.0) / 2.0  # for side=1
    
    # For each inner hexagon, find the maximum "extent" in the outer hexagon's frame
    # The outer hexagon contains a point if the point is within distance outer_inr * S
    # from the center along each of the 6 face-normal directions.
    
    # Transform all inner hexagon vertices to outer frame
    max_extent = 0.0
    
    oa = np.radians(-outer_angle)
    cos_oa = np.cos(oa)
    sin_oa = np.sin(oa)
    
    for i in range(N):
        cx, cy = positions[i]
        a = angles[i]
        verts = hex_vertices(cx, cy, a)
        
        # Transform to outer frame
        dx = verts[:, 0] - outer_center[0]
        dy = verts[:, 1] - outer_center[1]
        lx = dx * cos_oa - dy * sin_oa
        ly = dx * sin_oa + dy * cos_oa
        
        # Face normals in outer frame: 30, 90, 150, 210, 270, 330
        face_angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
        for fa in face_angles:
            proj = lx * np.cos(fa) + ly * np.sin(fa)
            max_extent = max(max_extent, np.max(proj))
    
    # The outer inradius must be at least max_extent
    outer_inradius_needed = max_extent
    # side = inradius / (sqrt(3)/2)
    side = outer_inradius_needed / (np.sqrt(3.0) / 2.0)
    return side

def objective(x):
    """
    Objective function: minimize outer hexagon side length.
    x: flattened array [cx1, cy1, a1, cx2, cy2, a2, ..., cxN, cyN, aN, ocx, ocy, oa]
    """
    N = 11
    positions = x[0:3*N:3]
    cys = x[1:3*N:3]
    angles = x[2:3*N:3]
    
    outer_center = x[3*N:3*N+2]
    outer_angle = x[3*N+2]
    
    pos = np.column_stack([positions, cys])
    
    # Check hexagon-hexagon separation
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = (positions[i], cys[i], angles[i])
            v2 = (positions[j], cys[j], angles[j])
            d = hex_hex_distance(v1, v2)
            if d < 0:
                penalty += d * d * 1000  # Quadratic penalty for overlap
    
    side = compute_outer_side(pos, angles, outer_center, outer_angle)
    
    return side + penalty

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Initial configurations
    best_side = np.inf
    best_x = None
    
    # Strategy 1: Flower pattern (1 center + 6 around + 4 more)
    # Unit hexagons: adjacent centers are 2*inradius = sqrt(3) apart when sharing an edge
    # But for disjoint (non-overlapping), they can touch, so center distance >= 2*inradius = sqrt(3)
    
    inr = np.sqrt(3.0) / 2.0
    center_dist = 2 * inr  # = sqrt(3)
    
    # Flower: center at origin, 6 around at distance center_dist
    # Additional 4: place them in the gaps
    
    configs = []
    
    # Config 1: Standard flower
    pos1 = [[0.0, 0.0]]
    for k in range(6):
        ang = np.radians(k * 60)
        pos1.append([center_dist * np.cos(ang), center_dist * np.sin(ang)])
    # Add 4 more: place at distance 2*center_dist in alternating directions
    for k in range(0, 6, 2):
        ang = np.radians(k * 60 + 30)
        pos1.append([2 * center_dist * np.cos(ang), 2 * center_dist * np.sin(ang)])
    # That's 11
    configs.append((np.array(pos1), np.zeros(11), np.array([0.0, 0.0]), 0.0))
    
    # Config 2: Linear arrangement
    pos2 = [[k * center_dist, 0.0] for k in range(11)]
    configs.append((np.array(pos2), np.zeros(11), np.array([0.0, 0.0]), 0.0))
    
    # Config 3: Two rows
    pos3 = []
    for k in range(6):
        pos3.append([k * center_dist, 0.0])
    for k in range(5):
        pos3.append([(k + 0.5) * center_dist, center_dist * np.sqrt(3)/2 * 2])
    configs.append((np.array(pos3), np.zeros(11), np.array([0.0, 0.0]), 0.0))
    
    # Config 4: Random
    np.random.seed(42)
    for trial in range(5):
        pos_r = np.random.uniform(-5, 5, (11, 2))
        ang_r = np.random.uniform(0, 60, 11)
        oc_r = np.random.uniform(-1, 1, 2)
        oa_r = np.random.uniform(0, 60)
        configs.append((pos_r, ang_r, oc_r, oa_r))
    
    # Optimize each config
    for idx, (pos, ang, oc, oa) in enumerate(configs):
        if time.time() > DEADLINE - 5:
            break
        
        x0 = np.concatenate([pos.flatten(), ang, oc, [oa]])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.success or result.fun < best_side:
                x_opt = result.x
                side_opt = compute_outer_side(
                    np.column_stack([x_opt[0:33:3], x_opt[1:33:3]]),
                    x_opt[2:33:3],
                    x_opt[33:35],
                    x_opt[35]
                )
                # Check for overlaps
                has_overlap = False
                for i in range(11):
                    for j in range(i+1, 11):
                        v1 = (x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2])
                        v2 = (x_opt[3*j], x_opt[3*j+1], x_opt[3*j+2])
                        d = hex_hex_distance(v1, v2)
                        if d < -1e-6:
                            has_overlap = True
                if not has_overlap and side_opt < best_side:
                    best_side = side_opt
                    best_x = x_opt
        except Exception:
            continue
    
    # If no valid solution found, use the flower pattern directly
    if best_x is None:
        # Use the flower config directly
        pos = np.array(configs[0][0])
        ang = configs[0][1]
        oc = configs[0][2]
        oa = configs[0][3]
        best_side = compute_outer_side(pos, ang, oc, oa)
        best_x = np.concatenate([pos.flatten(), ang, oc, [oa]])
    
    # Extract results
    positions = np.column_stack([best_x[0:33:3], best_x[1:33:3]])
    angles = best_x[2:33:3]
    outer_center = best_x[33:35]
    outer_angle = best_x[35]
    
    # Build inner list
    inner = []
    for i in range(11):
        inner.append([float(positions[i][0]), float(positions[i][1]), float(angles[i])])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(best_side), float(outer_angle)
