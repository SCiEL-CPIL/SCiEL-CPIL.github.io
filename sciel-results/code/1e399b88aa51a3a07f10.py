import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def sat_distance(v1, v2, tol=1e-9):
    """Compute minimum separation distance between two convex polygons using SAT.
    Returns positive if separated, 0 if touching, negative if overlapping."""
    # For two convex polygons, check all edges of both
    min_dist = np.inf
    
    # Get all edges from both polygons
    edges = []
    for poly in [v1, v2]:
        for i in range(len(poly)):
            edges.append(poly[i])
            edges.append(poly[(i+1) % len(poly)])
    
    # For each edge, compute the projection of all vertices
    for i in range(len(v1)):
        e1 = v1[(i+1) % 6] - v1[i]
        # Normal to edge
        n = np.array([-e1[1], e1[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        
        # Project all vertices onto this normal
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 - tol:
            return min2 - max1  # Separated
        if max2 < min1 - tol:
            return min1 - max2  # Separated
        # Overlapping on this axis, track minimum overlap
    
    for i in range(len(v2)):
        e1 = v2[(i+1) % 6] - v2[i]
        n = np.array([-e1[1], e1[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol:
            return min2 - max1
        if max2 < min1 - tol:
            return min1 - max2
    
    # If we get here, they overlap. Compute minimum penetration depth
    # Use a simpler approach: compute minimum distance between edges
    min_dist = np.inf
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1) % 6]
            # Distance between two line segments
            d = seg_seg_dist(p1, p2, p3, p4)
            min_dist = min(min_dist, d)
    
    return -min_dist  # Negative means overlap

def seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-15 and e < 1e-15:
        return np.linalg.norm(r)
    if a < 1e-15:
        t = f / e
        t = np.clip(t, 0, 1)
        return np.linalg.norm(r + t * d2)
    if e < 1e-15:
        t = -np.dot(d1, r) / a
        t = np.clip(t, 0, 1)
        return np.linalg.norm(d1 * t - r)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0, 1)
    t = (b * s + f) / e
    t = np.clip(t, 0, 1)
    
    if t < 1e-9:
        t = 0
        s = np.clip(-np.dot(d1, r) / a, 0, 1)
    elif t > 1 - 1e-9:
        t = 1
        s = np.clip((b - a) / a, 0, 1)
    elif s < 1e-9:
        s = 0
        t = np.clip(f / e, 0, 1)
    elif s > 1 - 1e-9:
        s = 1
        t = np.clip((b - a) / e, 0, 1)
    
    return np.linalg.norm(d1 * s - r - d2 * t)

def compute_objective(params, side_length):
    """Compute the objective: check if all hexagons fit inside the outer hexagon
    and don't overlap. Returns a penalty value."""
    cx, cy = params[0], params[1]
    outer_verts = hex_vertices(cx, cy, side_length, 0)
    
    total_penalty = 0.0
    
    # Get all inner hexagon vertices
    inner_verts = []
    for i in range(N):
        ix = params[2 + 3*i]
        iy = params[3 + 3*i]
        angle = params[4 + 3*i]
        verts = hex_vertices(ix, iy, 1.0, angle)
        inner_verts.append(verts)
    
    # Check each inner hexagon is inside outer hexagon
    for i in range(N):
        verts = inner_verts[i]
        for v in verts:
            # Check if point is inside outer hexagon
            inside = point_in_polygon(v, outer_verts)
            if not inside:
                # Compute distance to boundary
                d = point_to_polygon_dist(v, outer_verts)
                total_penalty += max(0, 0.1 - d) ** 2 * 100
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            # Use SAT-based check
            sep = sat_distance(v1, v2)
            if sep < 0:
                total_penalty += (-sep) ** 2 * 100
    
    return total_penalty

def point_in_polygon(point, polygon):
    """Check if point is inside convex polygon."""
    n = len(polygon)
    for i in range(n):
        v1 = polygon[i]
        v2 = polygon[(i+1) % n]
        cross = (v2[0]-v1[0])*(point[1]-v1[1]) - (v2[1]-v1[1])*(point[0]-v1[0])
        if cross < -1e-9:
            return False
    return True

def point_to_polygon_dist(point, polygon):
    """Minimum distance from point to polygon boundary (0 if inside)."""
    if point_in_polygon(point, polygon):
        return 0.0
    min_d = np.inf
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        d = point_to_seg_dist(point, p1, p2)
        min_d = min(min_d, d)
    return min_d

def point_to_seg_dist(point, p1, p2):
    """Distance from point to line segment."""
    d = p2 - p1
    a = np.dot(d, d)
    if a < 1e-15:
        return np.linalg.norm(point - p1)
    t = np.dot(point - p1, d) / a
    t = np.clip(t, 0, 1)
    return np.linalg.norm(point - (p1 + t * d))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: honeycomb-like arrangement
    # Place hexagons in a compact pattern
    # Center hexagon, then 6 around it, then 5 more
    init_params = [0.0, 0.0]  # outer center
    
    # Try a few different initial configurations
    best_score = np.inf
    best_params = None
    
    # Configuration 1: Linear
    for k in range(N):
        x = (k - N//2) * 1.75
        y = 0.0
        init_params.extend([x, y, 0.0])
    
    initial_sides = [4.0, 4.5, 5.0, 5.5, 6.0]
    
    for side in initial_sides:
        if time.time() > deadline - 30:
            break
        
        # Optimize for this side length
        result = _optimize(init_params, side, deadline)
        if result is not None:
            penalty = result[0]
            if penalty < best_score:
                best_score = penalty
                best_params = result[1]
    
    if best_params is None:
        # Fallback: return seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract parameters
    cx, cy = best_params[0], best_params[1]
    inner = []
    for i in range(N):
        ix = best_params[2 + 3*i]
        iy = best_params[3 + 3*i]
        angle = best_params[4 + 3*i]
        inner.append([ix, iy, angle])
    
    # Find minimum side length that works
    # Binary search on side length
    lo, hi = 2.0, 8.0
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) / 2
        params = [cx, cy] + best_params[2:]
        penalty = compute_objective(params, mid)
        if penalty < 1e-6:
            hi = mid
        else:
            lo = mid
    
    side = hi
    
    return inner, (cx, cy), side, 0.0

def _optimize(init_params, side_length, deadline):
    """Run SLSQP optimization."""
    try:
        from scipy.optimize import minimize
        
        def objective(params):
            return compute_objective(params, side_length)
        
        # Bounds
        bounds = [(-5, 5), (-5, 5)]
        for i in range(N):
            bounds.extend([(-5, 5), (-5, 5), (-180, 180)])
        
        result = minimize(
            objective,
            init_params,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success or result.fun < 1e-4:
            return (result.fun, result.x)
        else:
            return None
    except Exception:
        return None
