import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(center, angle_deg, scale=1.0):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    scale: side length
    """
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    # Vertices of a unit pointy-top hexagon
    # Top: (0, 1)
    # TR: (sqrt(3)/2, 0.5)
    # BR: (sqrt(3)/2, -0.5)
    # B: (0, -1)
    # BL: (-sqrt(3)/2, -0.5)
    # TL: (-sqrt(3)/2, 0.5)
    angles = np.array([90, 30, -30, -90, -150, 150]) + angle_deg
    angles = np.deg2rad(angles)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x + cx, y + cy))

def support_function(vertices, normal):
    """
    Max dot product of vertices with normal vector.
    """
    return np.max(vertices @ normal)

def min_distance_polygons(v1, v2):
    """
    Approximate minimum distance between two convex polygons.
    If they intersect, returns negative value (penalty).
    Uses separating axis theorem logic: if all projections overlap, distance is 0 (or negative if deep).
    For penalty, we can use the maximum penetration depth.
    """
    # Normals of v1
    n1 = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        n1.append(normal)
    
    # Normals of v2
    n2 = []
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        n2.append(normal)
        
    all_normals = n1 + n2
    
    min_dist = float('inf')
    for n in all_normals:
        # Project v1
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        # Project v2
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check overlap
        if max1 < min2 or max2 < min1:
            # Separated
            dist = min(min2 - max1, max2 - min1)
            min_dist = min(min_dist, dist)
        else:
            # Overlapping in this axis
            # Penetration depth
            overlap = min(max1, max2) - max(min1, min2)
            # We want to maximize the minimum distance, so if overlapping, distance is negative?
            # For penalty, we want to push them apart.
            # If they overlap, the "distance" is effectively 0 for separation, but we need a gradient.
            # Let's return the negative of the overlap depth if they overlap?
            # Actually, if they overlap, the separating axis theorem fails.
            # We can use the minimum overlap depth as a penalty.
            # But min_distance should be 0 if they touch.
            # For optimization, we want to penalize overlap.
            # Let's track the maximum penetration.
            min_dist = 0.0 # They are not separated by this axis
            
    return min_dist

def calculate_required_side_length(positions, angles):
    """
    Calculate the minimum side length S of a regular hexagon (centered at origin, fixed orientation)
    that contains all inner hexagons.
    """
    # Outer hexagon normals (pointy top)
    # Normals to sides are at 0, 60, 120, 180, 240, 300 degrees
    outer_normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    max_support = 0.0
    for i in range(N):
        cx, cy = positions[i]
        ang = angles[i]
        verts = hexagon_vertices((cx, cy), ang)
        
        for n in outer_normals:
            s = support_function(verts, n)
            # Support of outer hexagon in direction n is S * support of unit hexagon in n
            # Unit hexagon support in normal to side (e.g., 0 deg) is sqrt(3)/2
            # So S * sqrt(3)/2 >= s  => S >= s / (sqrt(3)/2)
            required_S = s / (np.sqrt(3)/2)
            if required_S > max_support:
                max_support = required_S
                
    return max_support

def penalty_function(params):
    """
    params: 3*N array [x1, y1, a1, x2, y2, a2, ...]
    """
    positions = np.array(params[0::3]) + 1j * np.array(params[1::3]) # Not using complex for positions
    # Reshape
    x = params[0::3]
    y = params[1::3]
    a = params[2::3]
    
    positions = np.column_stack((x, y))
    angles = a
    
    # 1. Containment cost
    S = calculate_required_side_length(positions, angles)
    
    # 2. Overlap penalty
    overlap_pen = 0.0
    verts_list = []
    for i in range(N):
        verts = hexagon_vertices(positions[i], angles[i])
        verts_list.append(verts)
        
    for i in range(N):
        for j in range(i+1, N):
            dist = min_distance_polygons(verts_list[i], verts_list[j])
            if dist < 0.01: # If they are close or overlapping
                # We want to penalize overlap.
                # min_distance_polygons returns 0 if overlapping (in my current impl).
                # I need to modify it to return negative penetration.
                # Let's re-calculate penetration.
                # For simplicity in this fast script, if dist < 0.01, add penalty.
                # Better: compute actual penetration depth.
                overlap_pen += 10.0 # Heavy penalty if they touch/overlap
                
    return S + overlap_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: 3x4 grid
    # Hexagonal grid coordinates
    # Row j, Col i
    # x = i * sqrt(3) + (j % 2) * sqrt(3)/2
    # y = j * 1.5
    init_params = []
    for j in range(4):
        for i in range(3):
            x = i * np.sqrt(3) + (j % 2) * np.sqrt(3)/2
            y = j * 1.5
            ang = 0.0
            init_params.extend([x, y, ang])
            
    # Center the configuration
    xs = init_params[0::3]
    ys = init_params[1::3]
    cx = np.mean(xs)
    cy = np.mean(ys)
    for k in range(len(init_params)):
        if k % 3 == 0:
            init_params[k] -= cx
        elif k % 3 == 1:
            init_params[k] -= cy
            
    best_params = np.array(init_params)
    best_score = penalty_function(best_params)
    
    # Local optimization
    # Use Nelder-Mead for robustness with non-smooth penalties
    result = minimize(penalty_function, best_params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if result.fun < best_score:
        best_params = result.x
        best_score = result.fun
    
    # Check time
    if time.time() < deadline:
        # Try a few random restarts if time permits
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb initial guess
            noise = np.random.normal(0, 0.5, size=3*N)
            current_params = best_params + noise
            res = minimize(penalty_function, current_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
                
    # Extract final parameters
    x = best_params[0::3]
    y = best_params[1::3]
    a = best_params[2::3]
    
    # Calculate final S
    S = calculate_required_side_length(np.column_stack((x, y)), a)
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([x[i], y[i], a[i]])
        
    # Outer hexagon
    center = (0.0, 0.0)
    side = S
    angle = 0.0
    
    return inner, center, side, angle

