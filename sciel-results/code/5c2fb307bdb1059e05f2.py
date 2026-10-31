import itertools
import numpy as np

def construct():
    # Generate hexagonal lattice points
    # Center
    points = [(0, 0)]
    
    # Ring 1 (6 points)
    # Axial coordinates for hex grid
    # Directions: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # Ring 1
    for d in dirs:
        points.append((d[0], d[1]))
        
    # Ring 2 (12 points)
    # To generate ring 2, we step 2 units in each of the 6 directions and fill in between?
    # Or simply iterate all axial coords (q, r) such that max(|q|, |r|, |q+r|) == 2
    ring2_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance is max(|q|, |r|, |q+r|)
            if max(abs(q), abs(r), abs(q+r)) == 2:
                ring2_points.append((q, r))
                
    # Convert axial to Cartesian
    # x = q + r/2
    # y = sqrt(3)/2 * r
    sqrt3 = np.sqrt(3)
    
    cart_points = []
    # Add center
    cart_points.append((0.0, 0.0))
    
    # Add ring 1
    for q, r in dirs:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        cart_points.append((x, y))
        
    # Add ring 2
    for q, r in ring2_points:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        cart_points.append((x, y))
        
    # cart_points has 19 points
    # We need to select 16 points to minimize the max pairwise distance
    # The 12 outer points are indices 7 to 18 (0-based: 0 is center, 1-6 ring1, 7-18 ring2)
    outer_indices = list(range(7, 19))
    
    best_min_max_dist = float('inf')
    best_subset = None
    
    # Try all combinations of removing 3 points from the 12 outer points
    # This leaves 1 center + 6 ring1 + 9 ring2 = 16 points
    for remove_combo in itertools.combinations(outer_indices, 3):
        remove_set = set(remove_combo)
        current_points = []
        for i in range(19):
            if i not in remove_set:
                current_points.append(cart_points[i])
                
        # Calculate max pairwise distance
        max_dist = 0.0
        n = len(current_points)
        for i in range(n):
            for j in range(i+1, n):
                dx = current_points[i][0] - current_points[j][0]
                dy = current_points[i][1] - current_points[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > max_dist:
                    max_dist = dist
                    
        if max_dist < best_min_max_dist:
            best_min_max_dist = max_dist
            best_subset = current_points
            
    # Return the best subset as numpy array
    return np.array(best_subset, dtype=float)
