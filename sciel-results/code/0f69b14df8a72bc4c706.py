import numpy as np
import time
import random
import math

N = 12
SQRT3 = np.sqrt(3.0)
DEADLINE = 145.0

# Base vertices for a unit regular hexagon with side length 1
# Vertices at angles 0, 60, 120, 180, 240, 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_vertices(cx, cy, angle_rad):
    """
    Returns (6, 2) vertices for a unit hexagon centered at (cx, cy) with rotation angle_rad.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ BASE_VERTS.T).T + np.array([cx, cy])

def compute_min_gap(v1, v2):
    """
    Compute the minimum gap between two convex hexagons using SAT (Separating Axis Theorem).
    Returns a positive value if disjoint, 0 if touching, and a negative value (penetration depth approx) if overlapping.
    Note: True penetration depth is complex, but for optimization, a signed distance or a proxy that is negative on overlap and positive on separation works.
    A common proxy is: min over axes of (gap if gap > 0 else -distance_to_closest_point_on_axis).
    Actually, if they overlap, the gap is 0 for all axes. 
    To get a negative value for overlap, we can compute the minimum distance between the two polygons.
    If distance > 0, they are disjoint.
    If distance < 0 (which implies overlap), we return -distance.
    
    Let's compute the minimum distance between the two convex polygons.
    If they are disjoint, min_dist > 0.
    If they overlap, min_dist < 0 (defined as -penetration_depth).
    
    Efficient way:
    1. Check all axes from v1 and v2.
    2. For each axis, compute the projection intervals.
    3. If intervals are disjoint, the gap is the distance between intervals.
    4. If intervals overlap, the gap is 0 (on this axis).
    5. The minimum gap over all axes is the distance if disjoint.
    6. If the minimum gap is 0, they might be touching or overlapping.
    7. To distinguish, check if any vertex of v1 is inside v2. If yes, they overlap.
    8. If they overlap, we need a penalty. A good penalty is the minimum distance from a vertex of one poly to the edges of the other, but that's complex.
    9. Alternatively, if we detect overlap, we can return a negative value proportional to the "depth".
       A simple depth proxy: the minimum distance from the center of v1 to the boundary of v2 (if center is inside).
       Or, simply return -1.0 (or some constant) if overlap is detected. This creates a "wall" that the optimizer must climb.
       For gradient-free methods like Nelder-Mead or CMA-ES, a constant penalty for overlap is often sufficient to push the solution out, provided the penalty is large enough.
       Let's use: if disjoint, return min_gap. If overlapping, return -1.0.
    """
    min_gap = np.inf
    
    # Axes from v1
    for k in range(6):
        e = v1[(k+1)%6] - v1[k]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-10: continue
        n /= norm
        p1 = v1.dot(n)
        p2 = v2.dot(n)
        max1, min1 = np.max(p1), np.min(p1)
        max2, min2 = np.max(p2), np.min(p2)
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
            
        if gap < min_gap:
            min_gap = gap
            
    # Axes from v2
    for k in range(6):
        e = v2[(k+1)%6] - v2[k]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-10: continue
        n /= norm
        p1 = v1.dot(n)
        p2 = v2.dot(n)
        max1, min1 = np.max(p1), np.min(p1)
        max2, min2 = np.max(p2), np.min(p2)
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
            
        if gap < min_gap:
            min_gap = gap
            
    # If min_gap > 0, they are disjoint. Return min_gap.
    if min_gap > 1e-9:
        return min_gap
        
    # If min_gap <= 0, they are touching or overlapping.
    # Check for overlap using point-in-polygon.
    # If any vertex of v1 is inside v2, or any vertex of v2 is inside v1, they overlap.
    # We can use a fast check: is the center of v1 inside v2?
    # If yes, they definitely overlap.
    # If no, check vertices.
    
    def point_in_poly(pt, poly):
        x_pt, y_pt = pt
        inside = False
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%6]
            if ((y1 > y_pt) != (y2 > y_pt)) and (x_pt < (x2 - x1) * (y_pt - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside

    # Check center of v1
    c1 = np.mean(v1, axis=0)
    if point_in_poly(c1, v2):
        return -1.0
        
    # Check center of v2
    c2 = np.mean(v2, axis=0)
    if point_in_poly(c2, v1):
        return -1.0
        
    # Check vertices
    for v in v1:
        if point_in_poly(v, v2):
            return -1.0
    for v in v2:
        if point_in_poly(v, v1):
            return -1.0
            
    # If we get here, they are likely just touching or very close.
    # Return a small negative value to encourage separation?
    # Or 0? If they are touching, it's a valid configuration.
    # However, due to floating point, they might be slightly overlapping.
    # Let's return 0.0 to indicate "no penalty" but also "no separation".
    # Actually, if they are touching, the distance is 0.
    return 0.0

def objective(x):
    """
    x: flat array of length 3*N (x, y, angle_rad for each hexagon)
    Returns: outer side length + penalty * overlap_penalty
    """
    pos_x = x[0::3]
    pos_y = x[1::3]
    ang = x[2::3]
    
    # 1. Compute Outer Side Length
    # Get all vertices
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = get_vertices(pos_x[i], pos_y[i], ang[i])
        
    x_coords = all_verts[:, 0]
    y_coords = all_verts[:, 1]
    
    # The outer hexagon is defined by 6 half-planes.
    # We want to find the minimum side length S such that all vertices are inside the hexagon.
    # A regular hexagon centered at (0,0) with side length S has vertices at distance S from center.
    # The distance from center to each side (apothem) is S * sqrt(3)/2.
    # The condition for a point (x,y) to be inside a regular hexagon with side S and orientation 0 is:
    # |x| <= S
    # |y| <= S * sqrt(3)/2
    # |x/2 + y*sqrt(3)/2| <= S
    # |x/2 - y*sqrt(3)/2| <= S
    # |y| <= S * sqrt(3)/2 (redundant)
    # Wait, let's use the support function approach.
    # The side length S is determined by the maximum projection of the points onto the 6 normal directions of the hexagon sides.
    # The normals are at angles 0, 60, 120, 180, 240, 300.
    # Let's define the 6 unit normals.
    normals = np.column_stack([BASE_COS, BASE_SIN])
    
    # Project all points onto each normal
    projections = all_verts.dot(normals.T) # (N*6, 6)
    
    # For each normal direction, the maximum projection is the "radius" in that direction.
    # The side length S must satisfy S >= max_proj / (apothem_factor).
    # For a regular hexagon with side S, the apothem (distance from center to side) is h = S * sqrt(3)/2.
    # So S >= max_proj / h.
    max_projections = np.max(projections, axis=0) # (6,)
    
    # We need to consider the absolute value because the hexagon is centered at 0.
    # Actually, the projection can be negative. The constraint is -h <= proj <= h.
    # So |proj| <= h.
    # Thus S >= max(|proj|) / h.
    abs_max_projections = np.max(np.abs(projections), axis=0)
    
    h = SQRT3 / 2.0
    S = np.max(abs_max_projections) / h
    
    # 2. Compute Overlap Penalty
    penalty = 0.0
    for i in range(N):
        v1 = all_verts[i*6:(i+1)*6]
        for j in range(i+1, N):
            v2 = all_verts[j*6:(j+1)*6]
            gap = compute_min_gap(v1, v2)
            if gap < 0:
                # Overlap detected. Add penalty.
                # The penalty should be large enough to dominate the side length.
                penalty += 1.0 * (-gap) # Since gap is -1, penalty += 1.
    
    return S + 10.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial Configuration: Honeycomb lattice
    # This is a valid, tight packing.
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_x = np.zeros(3*N)
    for i, (q, r) in enumerate(coords):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_x[3*i] = x
        init_x[3*i+1] = y
        init_x[3*i+2] = 0.0
        
    best_x = init_x.copy()
    best_val = objective(best_x)
    
    # Use Nelder-Mead for local refinement.
    # Nelder-Mead is derivative-free and robust to non-smooth functions (like our penalty).
    # We will run multiple restarts with different initial configurations if time permits.
    
    # Define a helper to run Nelder-Mead with a time check.
    def run_nm(x0, max_iter=500):
        nonlocal best_x, best_val, start_time
        # Simple time check in callback? 
        # SciPy's Nelder-Mead doesn't easily support early stopping via callback in all versions.
        # We'll just run it and hope it's fast.
        # Each function evaluation is O(N^2 * 6) ~ 864 ops.
        # 500 iterations * (N+1) evaluations ~ 500 * 13 = 6500 evals.
        # 6500 * 864 ~ 5.6M ops. Should be fast (< 1s).
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass

    # Run 1: From honeycomb
    run_nm(init_x)
    
    # Run 2: From a slightly perturbed honeycomb
    if time.time() - start_time < DEADLINE - 5:
        x2 = init_x + np.random.normal(0, 0.1, 3*N)
        run_nm(x2)
        
    # Run 3: From a different initial packing (e.g., 3x4 grid)
    if time.time() - start_time < DEADLINE - 5:
        # 3x4 grid of hexagons
        # This might not be as tight as honeycomb, but provides a different local minimum.
        coords_grid = []
        for r in range(4):
            for q in range(3):
                coords_grid.append((q, r))
                
        init_x_grid = np.zeros(3*N)
        for i, (q, r) in enumerate(coords_grid):
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            init_x_grid[3*i] = x
            init_x_grid[3*i+1] = y
            init_x_grid[3*i+2] = 0.0
            
        run_nm(init_x_grid)
        
    # Run 4: Random restart
    if time.time() - start_time < DEADLINE - 5:
        # Random positions in a circle of radius 5
        x3 = np.zeros(3*N)
        for i in range(N):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(0, 5)
            x3[3*i] = r * np.cos(angle)
            x3[3*i+1] = r * np.sin(angle)
            x3[3*i+2] = random.uniform(0, 2*np.pi)
        run_nm(x3)
        
    # Extract result
    inner = []
    for i in range(N):
        x = float(best_x[3*i])
        y = float(best_x[3*i+1])
        ang = float(np.degrees(best_x[3*i+2])) % 360.0
        inner.append([x, y, ang])
        
    # Recompute side length for the best valid config
    pos_x = best_x[0::3]
    pos_y = best_x[1::3]
    ang_rad = best_x[2::3]
    
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = get_vertices(pos_x[i], pos_y[i], ang_rad[i])
        
    x_coords = all_verts[:, 0]
    y_coords = all_verts[:, 1]
    
    normals = np.column_stack([BASE_COS, BASE_SIN])
    projections = all_verts.dot(normals.T)
    abs_max_projections = np.max(np.abs(projections), axis=0)
    h = SQRT3 / 2.0
    side_len = np.max(abs_max_projections) / h
    
    return inner, (0.0, 0.0), float(side_len), 0.0

# EVOLVE-BLOCK-END
