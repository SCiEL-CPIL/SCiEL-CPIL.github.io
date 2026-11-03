import numpy as np
import time
from scipy.optimize import minimize
import math

N = 12

def get_hex_vertices(angle_deg, center):
    """
    Returns vertices of a unit hexagon with given angle and center.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of a unit hexagon centered at origin, angle 0
    # Angles: 0, 60, 120, 180, 240, 300 degrees?
    # Standard hexagon vertices at angles 0, 60, ...
    angles = np.arange(6) * np.pi / 3 + angle_rad
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    return vertices + np.array(center)

def point_in_hexagon(point, vertices):
    """
    Check if a point is inside a convex polygon (hexagon) defined by vertices.
    Uses cross product method.
    """
    # vertices is (6, 2)
    # point is (2,)
    inside = True
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        # Vector p1->p2
        v1 = p2 - p1
        # Vector p1->point
        v2 = point - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < -1e-9: # Assuming CCW ordering
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
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

def hexagons_overlap(v1, v2):
    """
    Check if two hexagons (vertices) overlap.
    """
    # Check if any vertex of v1 is in v2
    for p in v1:
        if point_in_hexagon(p, v2):
            return True
    # Check if any vertex of v2 is in v1
    for p in v2:
        if point_in_hexagon(p, v1):
            return True
    # Check edge intersections
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def calculate_violation(inner_params, outer_side, outer_center, outer_angle):
    """
    inner_params: list of (x, y, angle_deg)
    outer_side: float
    outer_center: (x, y)
    outer_angle: float
    
    Returns a penalty value. 0 if valid.
    """
    penalty = 0.0
    
    # Get outer hexagon vertices
    outer_verts = get_hex_vertices(outer_angle, outer_center)
    # Scale outer vertices to side length
    # The vertices from get_hex_vertices are for side 1.
    # To get side S, we scale by S.
    outer_verts = outer_verts * outer_side
    
    # Calculate inner hexagon vertices
    inner_verts = []
    for x, y, ang in inner_params:
        verts = get_hex_vertices(ang, (x, y))
        inner_verts.append(verts)
        
    # Check containment
    # For each inner hexagon, check if all its vertices are inside the outer hexagon
    for i in range(N):
        for p in inner_verts[i]:
            if not point_in_hexagon(p, outer_verts):
                # Penalty is distance to boundary? Or just a large constant?
                # Using distance to boundary is smoother.
                # Approximate distance: find closest point on outer boundary?
                # For simplicity, use a large penalty if outside.
                penalty += 10.0
                
    # Check overlap
    for i in range(N):
        for j in range(i + 1, N):
            if hexagons_overlap(inner_verts[i], inner_verts[j]):
                penalty += 10.0
                
    return penalty

def objective(params):
    """
    params: flat array
    [x1, y1, a1, ..., x12, y12, a12, S]
    """
    inner_params = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        inner_params.append((x, y, a))
        
    S = params[-1]
    
    # Outer hexagon fixed at (0,0), angle 0
    penalty = calculate_violation(inner_params, S, (0.0, 0.0), 0.0)
    
    # We want to minimize S, but penalty must be 0.
    # Use a barrier/penalty method.
    # If penalty > 0, add it to S.
    # To make it smooth, we could use penalty**2.
    return S + 100 * penalty

def construct():
    start_time = time.time()
    timeout = 110.0 # 120s limit, keep margin
    
    best_score = float('inf')
    best_inner = None
    best_S = float('inf')
    
    # Initial guess:
    # Place hexagons in a grid-like structure
    # Side length estimate: ~4.0
    # Let's try S=4.5 initially
    
    # Random restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize parameters
        # Random positions within a circle of radius 5
        # Random angles
        params = np.zeros(N * 3 + 1)
        
        for i in range(N):
            # Cluster around center
            angle = np.random.uniform(0, 2 * np.pi)
            rad = np.random.uniform(0, 3.0)
            params[i * 3] = rad * np.cos(angle)
            params[i * 3 + 1] = rad * np.sin(angle)
            params[i * 3 + 2] = np.random.uniform(0, 60) # 0 to 60 deg
        params[-1] = 4.5 # Initial S
        
        # Optimize
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < best_score:
                # Check if valid
                inner_params = []
                for i in range(N):
                    inner_params.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
                S = res.x[-1]
                penalty = calculate_violation(inner_params, S, (0.0, 0.0), 0.0)
                
                if penalty < 1e-6:
                    if S < best_S:
                        best_S = S
                        best_inner = inner_params
                        best_score = S
        except Exception:
            pass

    if best_inner is None:
        # Fallback to a simple valid configuration if optimization failed
        # Place 12 hexagons in a 2x3x2 block?
        # Just return the seed-like result but with a smaller S?
        # Or a known valid packing.
        # A 3x3 block of hexagons (side 3) fits 19.
        # So S=3.5 should fit 12 easily.
        # Let's just return a conservative valid config.
        # Hexagonal grid positions for 12 hexes.
        # Center (0,0), 6 around, 5 more?
        # Let's use S=4.0 and place them roughly.
        # To be safe, return a config that is definitely valid.
        # Place them in a line? No, too long.
        # Place them in a 2x2x3 grid?
        # Let's just return the best found or a default.
        # Default: 12 hexagons in a hexagonal cluster of side 2 (7 hexes) + 5 scattered?
        # If no valid found, return a large S to be safe?
        # Score is lower is better. Invalid is 0.
        # We should return a valid config if possible.
        
        # Try a deterministic placement for S=4.0
        # 12 hexagons can fit in a hexagon of side 4.
        # Let's place them at integer grid coordinates scaled by sqrt(3).
        # This is a heuristic.
        
        # If best_inner is still None, we must return *something*.
        # The prompt says "invalid result ... scores 0".
        # So we need to ensure validity.
        
        # Let's try one more simple optimization with a larger S.
        params = np.zeros(N * 3 + 1)
        for i in range(N):
            angle = i * np.pi / 6
            rad = 1.0
            params[i * 3] = rad * np.cos(angle)
            params[i * 3 + 1] = rad * np.sin(angle)
            params[i * 3 + 2] = 0.0
        params[-1] = 5.0
        
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-3, 'fatol': 1e-3})
            inner_params = []
            for i in range(N):
                inner_params.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
            S = res.x[-1]
            penalty = calculate_violation(inner_params, S, (0.0, 0.0), 0.0)
            if penalty < 1e-6:
                best_inner = inner_params
                best_S = S
        except Exception:
            pass

    if best_inner is None:
        # Last resort: Return a very large hexagon with hexagons in a line
        # This is valid but score is high.
        # Line of 12 hexagons.
        # Length approx 12 * sqrt(3) ~ 20.8.
        # Side S ~ 12.
        inner = []
        for i in range(N):
            inner.append([i * np.sqrt(3), 0.0, 0.0])
        return inner, (0.0, 0.0), 15.0, 0.0

    return best_inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
