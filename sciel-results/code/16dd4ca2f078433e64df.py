import numpy as np
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Lattice spacing
    d = np.sqrt(3)
    
    # Generate lattice points
    # Basis vectors
    u = np.array([d, 0])
    v = np.array([d/2, 1.5])
    
    points = []
    # Generate a grid of points around the origin
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * u + j * v
            points.append(p)
            
    points = np.array(points)
    
    best_L = float('inf')
    best_points = None
    
    # We need to select 11 points.
    # To keep it fast, we can just pick the 11 closest to the origin?
    # Or search for the best subset.
    # Given N=11, checking all subsets is impossible.
    # Heuristic: The optimal packing is likely a "convex" subset of the lattice.
    # Let's try all subsets of size 11 from a small neighborhood.
    
    # Sort points by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx_sorted = np.argsort(dists)
    
    # Consider only the top 20 closest points to limit combinations
    candidate_indices = idx_sorted[:20]
    
    # Iterate through combinations of 11 from these 20
    # C(20, 11) is ~ 167960, which is manageable.
    for combo in itertools.combinations(candidate_indices, 11):
        subset = points[list(combo)]
        
        # Calculate minimal enclosing hexagon side L for this subset
        # The hexagon is aligned with the lattice.
        # We need to find the center (cx, cy) that minimizes L.
        # For a fixed orientation, the minimal L for a set of points is determined by the 
        # maximum projection onto the 6 normals.
        # The optimal center is the center of the minimal bounding box in the hexagonal metric.
        # Approximation: Use the centroid.
        center = np.mean(subset, axis=0)
        
        # Calculate L required to contain all points with this center
        # Normals for the 6 sides of a hexagon aligned with x-axis
        # Side 1: x <= L
        # Side 2: x >= -L
        # Side 3: y <= L*sqrt(3)/2
        # Side 4: y >= -L*sqrt(3)/2
        # Side 5: x + y/sqrt(3) <= L
        # Side 6: x - y/sqrt(3) >= -L
        
        rel = subset - center
        
        # Projections
        # The distance from center to the side is L * sqrt(3)/2
        # So L = 2/sqrt(3) * max_projection
        
        # Normal vectors (unit)
        # n1 = (1, 0) -> proj = x. Dist = x. L = 2/sqrt(3) * x
        # n2 = (-1, 0) -> proj = -x. Dist = -x. L = 2/sqrt(3) * -x
        # n3 = (0, 1) -> proj = y. Dist = y. L = 2/sqrt(3) * y / (sqrt(3)/2) ? No.
        # Apothem = L * sqrt(3)/2.
        # Distance to side 3 (top) is y. So y <= L * sqrt(3)/2.
        # L >= 2 * y / sqrt(3).
        
        L_x = 2 * np.max(np.abs(rel[:, 0])) / np.sqrt(3)
        L_y = 2 * np.max(np.abs(rel[:, 1])) / np.sqrt(3)
        
        # Diagonals
        # Side 5: x + y/sqrt(3) <= L
        # proj = x + y/sqrt(3).
        # L >= proj.
        proj_diag1 = rel[:, 0] + rel[:, 1] / np.sqrt(3)
        proj_diag2 = rel[:, 0] - rel[:, 1] / np.sqrt(3)
        
        L_d1 = np.max(np.abs(proj_diag1))
        L_d2 = np.max(np.abs(proj_diag2))
        
        L = max(L_x, L_y, L_d1, L_d2)
        
        if L < best_L:
            best_L = L
            best_points = subset
            best_center = center

    # Construct the result
    inner = []
    for p in best_points:
        inner.append([p[0], p[1], 0.0])
    
    outer_side = best_L + 1.0
    outer_center = [best_center[0], best_center[1]]
    
    return inner, outer_center, outer_side, 0.0
# EVOLVE-BLOCK-END
