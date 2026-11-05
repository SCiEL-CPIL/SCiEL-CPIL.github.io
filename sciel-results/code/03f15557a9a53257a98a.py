import math

def construct():
    n = 32
    
    # Define neighbors in axial coordinates (a, b)
    # The 6 neighbors of (0,0) are:
    # (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    # Generate candidate points in a box
    # A cluster of 32 points will have a radius of roughly sqrt(32/ (pi/sqrt(3)/2)) ~ 5
    # So a box of [-10, 10] is safe.
    max_coord = 10
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            candidates.append((a, b))
            
    selected = []
    selected_set = set()
    
    # Greedy selection
    for _ in range(n):
        best_point = None
        max_connections = -1
        # Tie-breaker: prefer points closer to origin to keep cluster compact
        # We can just iterate and keep the first one with max connections, 
        # but initializing with a specific point helps.
        
        # Optimization: Only consider points that are "close" to the current cluster?
        # For n=32, iterating over ~400 candidates 32 times is 12800 checks.
        # Each check involves 6 neighbor lookups in a set.
        # Total ops ~ 100k. Very fast.
        
        for p in candidates:
            if p in selected_set:
                continue
                
            # Count neighbors in selected_set
            connections = 0
            for da, db in neighbors:
                if (p[0] + da, p[1] + db) in selected_set:
                    connections += 1
            
            # If we have no selected points yet, connections will be 0 for all.
            # We need a tie-breaker to start the cluster.
            # Let's prefer (0,0) first, then points with smaller hex distance.
            if connections > max_connections:
                max_connections = connections
                best_point = p
            elif connections == max_connections:
                # Tie-breaker: smaller hex distance to origin
                # hex_dist(a, b) = (abs(a) + abs(b) + abs(a+b)) // 2
                if best_point is None:
                    best_point = p
                else:
                    dist_new = (abs(p[0]) + abs(p[1]) + abs(p[0] + p[1])) // 2
                    dist_old = (abs(best_point[0]) + abs(best_point[1]) + abs(best_point[0] + best_point[1])) // 2
                    if dist_new < dist_old:
                        best_point = p
        
        if best_point is None:
            break
            
        selected.append(best_point)
        selected_set.add(best_point)
        
    # Convert to list of lists
    result = [[a, b] for a, b in selected]
    
    return result
