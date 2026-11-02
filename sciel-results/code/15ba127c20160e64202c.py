import numpy as np
from scipy.optimize import differential_evolution, basinhopping
import time

N = 12

def get_hex_vertices(cx, cy, r, angle):
    # r is circumradius (1 for unit hex)
    # angle is rotation
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle
    return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

def hexagon_distance(v1, v2):
    # Calculate minimum distance between two convex polygons (hexagons)
    # Using Separating Axis Theorem (SAT) for distance
    # If they intersect, distance is 0.
    # For optimization, we need a smooth penalty.
    # We can compute the minimum distance between any vertex of v1 and any edge of v2, and vice versa.
    # This is O(36) per pair. 66 pairs * 36 = ~2500 ops. Very fast.
    
    min_dist = np.inf
    # Distance from vertices of v1 to edges of v2
    for i in range(6):
        p1 = v1[i]
        for j in range(6):
            p2_start = v2[j]
            p2_end = v2[(j+1)%6]
            # Point to segment distance
            v = p2_end - p2_start
            w = p1 - p2_start
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p1 - p2_start)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(p1 - p2_end)
                else:
                    t = c1 / c2
                    proj = p2_start + t * v
                    d = np.linalg.norm(p1 - proj)
            if d < min_dist:
                min_dist = d
    # Distance from vertices of v2 to edges of v1
    for i in range(6):
        p1 = v2[i]
        for j in range(6):
            p2_start = v1[j]
            p2_end = v1[(j+1)%6]
            v = p2_end - p2_start
            w = p1 - p2_start
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p1 - p2_start)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    d = np.linalg.norm(p1 - p2_end)
                else:
                    t = c1 / c2
                    proj = p2_start + t * v
                    d = np.linalg.norm(p1 - proj)
            if d < min_dist:
                min_dist = d
                
    # Check if they intersect (min_dist should be 0 if intersecting, but numerical issues might make it small positive)
    # A more robust check for intersection is better, but for penalty, small positive distance is a penalty.
    # If they intersect, distance is 0.
    # To make it a penalty, we want to push them apart.
    # If min_dist < epsilon, they are overlapping or touching.
    # We want min_dist >= 0.
    # Actually, if they intersect, the "distance" between sets is 0.
    # We need to measure the "penetration depth" or just use the fact that distance is 0.
    # A smooth penalty for non-overlap: max(0, required_dist - actual_dist)^2.
    # But for hexagons, "required_dist" is 0.
    # So penalty is 0 if min_dist > 0.
    # If min_dist == 0, penalty is 0? No, they touch.
    # If they intersect, min_dist is 0 (mathematically).
    # We need to distinguish "touching" from "intersecting".
    # SAT can tell us if they intersect.
    
    # Let's use a simpler metric for the optimizer:
    # Compute the minimum distance between the two polygons.
    # If they intersect, this function returns 0.
    # We want them to be disjoint.
    # So we want min_dist > 0.
    # But we can't enforce min_dist > 0 strictly in a smooth way if it can be 0.
    # We can use a penalty: max(0, -min_dist + epsilon)^2? No.
    
    # Better approach:
    # Calculate the minimum distance between the two polygons.
    # If they are disjoint, d > 0.
    # If they touch, d = 0.
    # If they intersect, d = 0 (but we know they intersect via SAT).
    
    # Let's implement a robust intersection check and distance.
    # For the penalty, we can use the "minimum separation" which is 0 if intersecting/touching.
    # To encourage separation, we can penalize if min_dist < 0.01.
    # But if they intersect, min_dist is 0.
    
    # Let's use the area of intersection? Too slow.
    
    # Let's stick to:
    # 1. Check intersection using SAT.
    # 2. If intersect, penalty = 1.0 (or large).
    # 3. If not intersect, penalty = max(0, 0.01 - min_dist)^2.
    
    # Actually, for DE, non-smooth is okay.
    pass

def check_intersections(v1, v2):
    # Returns True if polygons intersect
    # SAT for two convex polygons
    # Axes are normals to edges of both polygons
    def project_polygon(vertices, axis):
        dots = np.dot(vertices, axis)
        return min(dots), max(dots)
    
    axes = []
    for i in range(6):
        # Edge v1[i] to v1[i+1]
        edge = v1[(i+1)%6] - v1[i]
        # Normal
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
    for i in range(6):
        edge = v2[(i+1)%6] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        
    for axis in axes:
        min1, max1 = project_polygon(v1, axis)
        min2, max2 = project_polygon(v2, axis)
        if max1 < min2 or max2 < min1:
            return False # Separating axis found
    return True # No separating axis, so they intersect

def objective(params):
    # params: [x1, y1, a1, ..., x12, y12, a12, S]
    S = params[-1]
    coords = params[:-1].reshape(12, 3)
    
    penalty = 0.0
    
    # 1. Containment
    # Outer hexagon vertices
    outer_verts = get_hex_vertices(0, 0, S, 0)
    
    for i in range(12):
        x, y, a = coords[i]
        inner_verts = get_hex_vertices(x, y, 1.0, a)
        
        # Check if all inner verts are inside outer hex
        # Use point-in-polygon or half-plane check
        # Half-plane check is faster
        # Outer hex sides:
        # Since outer hex is centered at 0,0 and aligned, its supporting lines are:
        # x * cos(30) + y * sin(30) <= S * sqrt(3)/2  ? No.
        # Distance from center to side is inradius R_in = S * sqrt(3)/2.
        # The 6 normals are at angles 30, 90, 150, 210, 270, 330.
        # For each normal n, dot(p, n) <= R_in.
        
        R_in = S * np.sqrt(3) / 2
        for k in range(6):
            angle = np.pi/6 + k * np.pi/3
            nx = np.cos(angle)
            ny = np.sin(angle)
            for vx, vy in inner_verts:
                dist = vx*nx + vy*ny
                if dist > R_in:
                    penalty += (dist - R_in)**2
    
    # 2. Overlap
    for i in range(12):
        v1 = get_hex_vertices(coords[i,0], coords[i,1], 1.0, coords[i,2])
        for j in range(i+1, 12):
            v2 = get_hex_vertices(coords[j,0], coords[j,1], 1.0, coords[j,2])
            if check_intersections(v1, v2):
                penalty += 1.0
            else:
                # Add small penalty if too close to prevent numerical sticking
                # Calculate min dist
                # Re-use logic from hexagon_distance or just skip for speed
                pass
                
    return S + 10.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial guess
    # 19-hex cluster subset
    # Center (0,0)
    # 6 neighbors at dist sqrt(3)
    # 5 more from ring 2
    
    # Let's just use random initial guess for DE
    bounds = [(-2.5, 2.5), (-2.5, 2.5), (0, np.pi/3)] * 12 + [(3.464, 4.0)]
    
    # Run DE
    result = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-6, polish=False)
    
    best_params = result.x
    best_score = result.fun
    
    # Refine with Basin Hopping
    # Define a local optimizer
    def local_obj(x):
        return objective(x)
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B needs gradients, we don't have them. Use Nelder-Mead.
    # But Nelder-Mead might be slow.
    # Let's just do a few more DE iterations if time permits.
    
    elapsed = time.time() - start_time
    if elapsed < 120:
         # Run another round of DE with tighter bounds around best_params
         # Or just return best_params
         pass
         
    # Extract results
    S = best_params[-1]
    inner = []
    for i in range(12):
        x, y, a = best_params[i*3 : i*3+3]
        inner.append([x, y, np.degrees(a)])
        
    return inner, (0.0, 0.0), S, 0.0
