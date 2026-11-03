import itertools
import math

N = 12

def construct():
    # Lattice vectors
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2.0, 1.5)
    
    # Generate points with hex distance <= 2
    # Hex distance d(q, r) = (|q| + |r| + |q+r|) / 2
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            d = (abs(q) + abs(r) + abs(q + r)) / 2.0
            if d <= 2.0:
                x = q * u[0] + r * v[0]
                y = q * u[1] + r * v[1]
                points.append((x, y))
                
    # There are 19 points
    # We need to choose 12
    
    best_L = float('inf')
    best_points = None
    
    # Normals for the 3 axes of the hexagon (aligned)
    # Normal 1: (0, 1) -> y-axis
    # Normal 2: (sqrt(3)/2, -1/2) -> 30 deg from x-axis? 
    # Let's use the standard 3 directions: 90, 150, 210 degrees.
    # n0 = (0, 1)
    # n1 = (-sqrt(3)/2, -0.5)
    # n2 = (sqrt(3)/2, -0.5)
    # Actually, just use the 3 axes of the lattice.
    # Axis 1: y
    # Axis 2: 60 deg
    # Axis 3: 120 deg
    
    n0 = (0.0, 1.0)
    n1 = (sqrt3/2.0, 0.5) # 30 deg? No, 60 deg is (0.5, sqrt3/2). 
    # Let's stick to the projection on the normals of the hexagon sides.
    # Hexagon sides are at 30, 90, 150, 210, 270, 330.
    # Normals are at 90, 150, 210, 270, 330, 30.
    # Unique normals: 90, 150, 210.
    n_90 = (0.0, 1.0)
    n_150 = (-sqrt3/2.0, 0.5)
    n_210 = (-sqrt3/2.0, -0.5)
    
    normals = [n_90, n_150, n_210]
    
    # Iterate over all combinations of 12 points from the 19
    # 19 choose 12 is 50388
    for combo in itertools.combinations(points, N):
        # Center the points
        cx = sum(p[0] for p in combo) / N
        cy = sum(p[1] for p in combo) / N
        
        centered = [(p[0] - cx, p[1] - cy) for p in combo]
        
        max_proj = 0.0
        for n in normals:
            min_p = float('inf')
            max_p = float('-inf')
            for p in centered:
                proj = p[0]*n[0] + p[1]*n[1]
                if proj < min_p: min_p = proj
                if proj > max_p: max_p = proj
            # The width is max_p - min_p
            # The required L for this axis is width / sqrt(3)
            # But we are centering, so we look at max absolute projection?
            # If we center by mean, the hexagon center is not necessarily the mean.
            # The optimal center minimizes the max absolute projection.
            # For a convex symmetric set, mean is good.
            # Let's just use the width.
            width = max_p - min_p
            L_candidate = width / sqrt3
            if L_candidate > max_proj:
                max_proj = L_candidate
                
        if max_proj < best_L:
            best_L = max_proj
            best_points = combo
            
    # Construct the output
    # We need to return the points, center, side, angle
    # The center we used was the mean of the points.
    # We should return the points relative to the origin (0,0) for the outer hexagon?
    # The interface says: inner rows (x, y, angle), outer center (x, y).
    # It's best to return the inner points in the coordinate system of the outer hexagon center.
    # So we can just return the centered points, and outer center (0,0).
    
    inner = [[p[0], p[1], 0.0] for p in best_points]
    # Wait, the points in best_points are the original lattice coords.
    # We need to center them.
    cx = sum(p[0] for p in best_points) / N
    cy = sum(p[1] for p in best_points) / N
    inner = [[p[0] - cx, p[1] - cy, 0.0] for p in best_points]
    
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-END
