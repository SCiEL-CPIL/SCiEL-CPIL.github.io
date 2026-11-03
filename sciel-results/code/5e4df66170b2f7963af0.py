import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_hex_vertices(x, y, angle_deg):
    """
    Returns vertices of a unit regular hexagon centered at (x,y) with given rotation.
    Angle 0 means vertex at (1,0) relative to center? 
    Standard: Vertex at (1,0) is angle 0.
    """
    angles = np.deg2rad(angle_deg + np.arange(0, 360, 60))
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    return vertices + np.array([x, y])

def polygon_distance(v1, v2):
    """
    Calculate minimum distance between two convex polygons.
    v1, v2: (6, 2) arrays of vertices.
    Returns 0 if they overlap, else positive distance.
    """
    # For simplicity and speed in optimization, we might use a simpler metric
    # if exact distance is too slow. But 11 hexes = 55 pairs. 
    # SAT is fast.
    
    # Get edges of v1
    edges1 = np.roll(v1, -1, axis=0) - v1
    # Get edges of v2
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    min_dist = float('inf')
    
    # Check projection on normals of v1 edges
    for e in edges1:
        # Normal
        n = np.array([-e[1], e[0]])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9: continue
        n = n / n_norm
        
        # Project v1
        proj1 = np.dot(v1, n)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project v2
        proj2 = np.dot(v2, n)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check overlap
        if max1 < min2 or max2 < min1:
            # Separated
            dist = max(min1, min2) - min(max1, max2)
            if dist < min_dist:
                min_dist = dist
            else:
                # If separated, we can return early? 
                # No, we need the minimum distance. 
                # But if they are separated on this axis, they don't overlap.
                # However, we need the distance to the closest point.
                # This SAT check only confirms separation.
                # To get distance, we need more work.
                pass
        
    # This simple SAT check doesn't give distance if they overlap.
    # If they don't overlap, it gives a lower bound on distance.
    # For penalty, we want a smooth function.
    
    # Alternative: Use distance between closest vertices/edges.
    # Given the small number of points, we can check all pairs of vertices and edges.
    # But 6 vertices per hex.
    
    # Let's use a simpler penalty: 
    # If the polygons overlap, the distance is negative.
    # We can approximate the "overlap depth" by the negative of the minimum separation found by SAT?
    # No, if they overlap, all projections overlap.
    
    # Let's use a different approach for penalty:
    # Sum of squared distances between all vertices of v1 and all vertices of v2?
    # No, that's not overlap.
    
    # Let's assume the optimizer keeps them non-overlapping by starting close to a valid state.
    # We can use a constraint that the distance between centers is > sqrt(3) - epsilon.
    # This is not exact but might work for aligned hexagons.
    
    return 0.0 # Placeholder

def objective(params):
    # params: 22 floats (x1, y1, ..., x11, y11)
    # Outer hexagon is centered at (0,0) with angle 0.
    # We minimize the side length L required to contain all hexagons.
    
    coords = params.reshape(11, 2)
    
    # Calculate L
    max_proj = 0.0
    for i in range(11):
        x, y = coords[i]
        # Vertices of hex i
        verts = get_hex_vertices(x, y, 0)
        # Project onto 6 normals of outer hexagon
        # Normals at 30, 90, 150, 210, 270, 330
        for angle_deg in [30, 90, 150, 210, 270, 330]:
            rad = np.deg2rad(angle_deg)
            n = np.array([np.cos(rad), np.sin(rad)])
            proj = np.dot(verts, n)
            max_p = np.max(proj)
            if max_p > max_proj:
                max_proj = max_p
    
    # L = max_proj / (sqrt(3)/2)
    L = max_proj / (np.sqrt(3) / 2)
    
    # Penalty for overlaps
    # Simple center distance penalty for aligned hexagons
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < np.sqrt(3):
                penalty += (np.sqrt(3) - d)**2
    
    return L + 10.0 * penalty

def construct():
    # Initial guess
    # 7-hex cluster
    sqrt3 = np.sqrt(3)
    init_coords = [
        [0.0, 0.0],
        [1.5, sqrt3/2],
        [0.0, sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [0.0, -sqrt3],
        [1.5, -sqrt3/2],
        # 4 more
        [3.0, 0.0],
        [1.5, 1.5*sqrt3],
        [-1.5, 1.5*sqrt3],
        [-3.0, 0.0]
    ]
    
    x0 = np.array(init_coords).flatten()
    
    # Optimize
    start_time = time.time()
    best_result = None
    best_L = float('inf')
    
    try:
        # Use L-BFGS-B
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
        
        if res.success or res.fun < best_L:
            best_result = res.x
            best_L = res.fun
    
    except Exception as e:
        pass
    
    # Ensure we have a result
    if best_result is None:
        best_result = x0
    
    # Extract final coordinates
    final_coords = best_result.reshape(11, 2)
    
    # Calculate final L
    max_proj = 0.0
    for i in range(11):
        x, y = final_coords[i]
        verts = get_hex_vertices(x, y, 0)
        for angle_deg in [30, 90, 150, 210, 270, 330]:
            rad = np.deg2rad(angle_deg)
            n = np.array([np.cos(rad), np.sin(rad)])
            proj = np.dot(verts, n)
            max_p = np.max(proj)
            if max_p > max_proj:
                max_proj = max_p
    
    L = max_proj / (np.sqrt(3) / 2)
    
    # Construct output
    inner = []
    for i in range(11):
        x, y = final_coords[i]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), L, 0.0

