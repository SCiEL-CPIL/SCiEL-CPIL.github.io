import math
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Directions on triangular lattice
    # (da, db) such that distance is 1
    # a^2 + b^2 + ab = 1
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # 1. Generate points in R <= 2 (Hexagon of radius 2)
    # BFS from (0,0)
    visited = set()
    layers = {0: [(0,0)]}
    current_layer = [(0,0)]
    visited.add((0,0))
    
    for r in range(1, 3):
        next_layer = []
        for p in current_layer:
            a, b = p
            for da, db in dirs:
                np = (a+da, b+db)
                if np not in visited:
                    visited.add(np)
                    next_layer.append(np)
        layers[r] = next_layer
        current_layer = next_layer
    
    base_points = []
    for r in range(3):
        base_points.extend(layers[r])
        
    # 2. Generate points in R=3
    next_layer = []
    for p in current_layer:
        a, b = p
        for da, db in dirs:
            np = (a+da, b+db)
            if np not in visited:
                visited.add(np)
                next_layer.append(np)
    ring3 = next_layer
    # ring3 should have 18 points
    
    # 3. Determine the cyclic order of ring3
    # We can sort by angle to get the cycle order
    def angle(p):
        a, b = p
        # Convert to Cartesian for angle
        # x = a + b/2, y = b*sqrt(3)/2
        x = a + 0.5*b
        y = math.sqrt(3)/2 * b
        return math.atan2(y, x)
        
    ring3_sorted = sorted(ring3, key=angle)
    
    # 4. For each point in ring3, count neighbors in base_points
    base_set = set(base_points)
    neighbor_counts = {}
    for p in ring3:
        a, b = p
        count = 0
        for da, db in dirs:
            if (a+da, b+db) in base_set:
                count += 1
        neighbor_counts[p] = count
        
    # 5. Find the best contiguous block of 13 in ring3_sorted
    n_ring = len(ring3_sorted)
    k = 13
    best_score = -1
    best_start = 0
    
    # Precompute prefix sums of neighbor counts? 
    # Since it's a cycle, we can just iterate.
    # Score = sum(neighbor_counts) + (k-1) edges internal
    # Internal edges are always k-1 if contiguous.
    # So we just maximize sum(neighbor_counts).
    
    for i in range(n_ring):
        s = 0
        for j in range(k):
            idx = (i + j) % n_ring
            s += neighbor_counts[ring3_sorted[idx]]
        if s > best_score:
            best_score = s
            best_start = i
            
    # Construct final set
    selected_ring3 = []
    for j in range(k):
        idx = (best_start + j) % n_ring
        selected_ring3.append(ring3_sorted[idx])
        
    final_points = base_points + selected_ring3
    
    # Verify count
    if len(final_points) != 32:
        # Fallback to simple spiral if something went wrong
        # But this should be 19 + 13 = 32
        pass
        
    # Verify constraints and count edges
    # Convert to list of [a,b]
    result = [[a, b] for a, b in final_points]
    
    # Check time
    if time.time() > deadline - 5:
         return result
         
    return result
