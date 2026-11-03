import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the equilateral triangle
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])
CENTROID = (V1 + V2 + V3) / 3.0

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by N points.
    points: (N, 2) array.
    Returns: float, the minimum area.
    Uses vectorized computation for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate indices for all unique triples (i, j, k) with i < j < k
    # This avoids loops in Python for the main calculation
    # C(11, 3) = 165 triples.
    # We can construct indices efficiently.
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Vectors
    v_ij = pj - pi
    v_ik = pk - pi
    
    # Cross product in 2D: x1*y2 - y1*x2
    cross = v_ij[:, 0] * v_ik[:, 1] - v_ij[:, 1] * v_ik[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _is_inside_triangle(p, v1, v2, v3):
    """
    Check if point p is inside triangle v1, v2, v3.
    Returns True if inside or on boundary.
    """
    # Barycentric coordinates
    x, y = p
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-10:
        return False
        
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c = 1 - a - b
    
    # Allow slight floating point error
    return a >= -1e-9 and b >= -1e-9 and c >= -1e-9

def _project_to_triangle(p, v1, v2, v3):
    """
    Project point p to the triangle if it is outside.
    Simple projection: if outside, move towards centroid until inside.
    This is a crude projection but sufficient for repair.
    """
    if _is_inside_triangle(p, v1, v2, v3):
        return p
    
    # Binary search for the closest point to p on the segment from p to centroid
    centroid = (v1 + v2 + v3) / 3.0
    lo = 0.0
    hi = 1.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        trial = p + mid * (centroid - p)
        if _is_inside_triangle(trial, v1, v2, v3):
            hi = mid
        else:
            lo = mid
    return p + hi * (centroid - p)

def _objective(points):
    """
    Objective function for minimization.
    We want to MAXIMIZE the min area, so we MINIMIZE -min_area.
    We add a large penalty if points are outside the triangle.
    """
    # Check bounds
    penalty = 0.0
    for p in points:
        if not _is_inside_triangle(p, V1, V2, V3):
            penalty += 1000.0
            
    if penalty > 0:
        return penalty
        
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_area = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start Nelder-Mead with diverse initializations
    # 1. Structured grid in barycentric coords
    # 2. Random uniform
    # 3. Random Gaussian
    # 4. Hexagonal packing
    
    # Pre-compute some structured initializations
    inits = []
    
    # 1. Grid in barycentric coordinates
    # Barycentric coords (u, v) where w = 1 - u - v
    # u, v >= 0, u + v <= 1
    # Map to cartesian: P = u*V1 + v*V2 + w*V3
    # We want 11 points.
    # A 3x3 grid in (u,v) has 9 points. We can add 2 more.
    # Or use a finer grid and pick 11.
    # Let's try a 4x4 grid subset.
    grid_u = [0.1, 0.3, 0.5, 0.7]
    grid_v = [0.1, 0.3, 0.5, 0.7]
    candidates = []
    for u in grid_u:
        for v in grid_v:
            if u + v <= 1.0:
                # Convert to cartesian
                w = 1.0 - u - v
                p = u * V1 + v * V2 + w * V3
                candidates.append(p)
                
    # We have more than 11 candidates? 
    # u=0.1, v=0.1 (0.2), v=0.3 (0.4), v=0.5 (0.6), v=0.7 (0.8) -> 4
    # u=0.3, v=0.1 (0.4), v=0.3 (0.6), v=0.5 (0.8) -> 3
    # u=0.5, v=0.1 (0.6), v=0.3 (0.8) -> 2
    # u=0.7, v=0.1 (0.8) -> 1
    # Total 10.
    # Add center?
    center_bary = np.array([1/3, 1/3])
    w = 1 - 2/3
    p_center = center_bary[0]*V1 + center_bary[1]*V2 + w*V3
    candidates.append(p_center)
    
    if len(candidates) >= N:
        # Pick 11 most spread out? Or just the first 11?
        # Let's just take the first 11.
        init_grid = np.array(candidates[:N])
        inits.append(init_grid)
        
    # 2. Random Uniform
    points = []
    for _ in range(N):
        while True:
            p = rng.uniform(0, 1, 2)
            if _is_inside_triangle(p, V1, V2, V3):
                points.append(p)
                break
    inits.append(np.array(points))
    
    # 3. Random Gaussian around centroid
    points = []
    for _ in range(N):
        while True:
            p = CENTROID + rng.normal(0, 0.2, 2)
            if _is_inside_triangle(p, V1, V2, V3):
                points.append(p)
                break
    inits.append(np.array(points))
    
    # 4. Hexagonal-ish
    center = CENTROID
    pts = [center]
    # 6 points around center
    for k in range(6):
        angle = np.pi / 3 * k
        r = 0.2
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        pts.append([x, y])
    # 4 more points near corners
    # V1, V2, V3 are corners.
    # Move slightly inwards
    pts.append(V1 + 0.1 * (CENTROID - V1))
    pts.append(V2 + 0.1 * (CENTROID - V2))
    pts.append(V3 + 0.1 * (CENTROID - V3))
    # One more random
    while True:
        p = rng.uniform(0, 1, 2)
        if _is_inside_triangle(p, V1, V2, V3):
            pts.append(p)
            break
    inits.append(np.array(pts))
    
    # Add a few random restarts
    for _ in range(5):
        points = []
        for _ in range(N):
            while True:
                p = rng.uniform(0, 1, 2)
                if _is_inside_triangle(p, V1, V2, V3):
                    points.append(p)
                    break
        inits.append(np.array(points))
        
    # Optimize each initialization
    for init_points in inits:
        if time.time() > deadline - 10.0:
            break
            
        current_points = init_points.copy()
        
        # Initial evaluation
        current_area = _min_triangle_area(current_points)
        if current_area > best_area:
            best_area = current_area
            best_points = current_points.copy()
            
        # Nelder-Mead optimization
        # We minimize _objective(current_points)
        # Flatten points to 1D array for scipy
        x0 = current_points.flatten()
        
        def obj_flat(x):
            pts = x.reshape(N, 2)
            return _objective(pts)
            
        try:
            res = minimize(obj_flat, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 1000, 'adaptive': True})
            
            if res.success:
                new_points = res.x.reshape(N, 2)
                # Verify inside
                valid = True
                for p in new_points:
                    if not _is_inside_triangle(p, V1, V2, V3):
                        valid = False
                        break
                if valid:
                    new_area = _min_triangle_area(new_points)
                    if new_area > best_area:
                        best_area = new_area
                        best_points = new_points.copy()
                        
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        best_points = inits[0]
        
    return best_points
