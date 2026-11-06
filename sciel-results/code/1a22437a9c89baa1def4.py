import math
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Define neighbors in the triangular lattice
    # (a, b) neighbors: (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate points in a hexagon of radius 3
    # a^2 + ab + b^2 <= 9
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a*a + a*b + b*b <= 9:
                points.append((a, b))
                
    # We have 37 points. We need 32.
    # We will remove 5 points.
    # To maximize remaining edges, we should remove points with the fewest connections to other points in the set.
    # This is a greedy approach.
    
    current_points = set(points)
    
    # Helper to get neighbors of a point within the current set
    def get_neighbors(p, point_set):
        a, b = p
        nbs = []
        for da, db in neighbors:
            np = (a+da, b+db)
            if np in point_set:
                nbs.append(np)
        return nbs
        
    # Calculate degrees
    degrees = {}
    for p in current_points:
        degrees[p] = len(get_neighbors(p, current_points))
        
    # Remove 5 points
    for _ in range(5):
        # Find point with minimum degree
        # Tie-breaker: remove point with larger distance from origin (more peripheral)
        min_deg = float('inf')
        max_dist = -1
        candidate = None
        for p in current_points:
            d = degrees[p]
            dist = p[0]*p[0] + p[0]*p[1] + p[1]*p[1]
            if d < min_deg or (d == min_deg and dist > max_dist):
                min_deg = d
                max_dist = dist
                candidate = p
                
        if candidate is not None:
            # Remove candidate
            current_points.remove(candidate)
            # Update degrees of neighbors
            for nb in get_neighbors(candidate, current_points):
                degrees[nb] -= 1
                
        if time.time() > deadline - 10:
            break
            
    # Convert to list of lists
    result = [[a, b] for a, b in current_points]
    
    # Verify count
    if len(result) != 32:
        # Fallback: just take first 32 from original list (should not happen)
        result = [[a, b] for a, b in points[:32]]
        
    return result
