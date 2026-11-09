"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Directions for the 6 neighbors on the triangular lattice
    # using axial coordinates (a, b) where point is a + b*omega
    # omega = exp(i*pi/3)
    # Neighbors of (a, b) are:
    # (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    points = []
    visited = set()
    queue = [(0, 0)]
    visited.add((0, 0))
    
    # We need exactly 32 points
    target_count = 32
    
    while len(points) < target_count:
        next_queue = []
        for a, b in queue:
            # Add current point to the result
            points.append([a, b])
            if len(points) >= target_count:
                break
            
            # Add unvisited neighbors to the next layer
            for da, db in dirs:
                na, nb = a + da, b + db
                if (na, nb) not in visited:
                    visited.add((na, nb))
                    next_queue.append((na, nb))
        
        if not next_queue:
            # This should not happen for n=32, but good for safety
            break
        queue = next_queue
    
    # Verify distinctness and bounds (though BFS from 0 with small n ensures this)
    # All points are distinct by construction (visited set)
    # Bounds: max coordinate magnitude is small (approx sqrt(32) or less)
    
    return points
# EVOLVE-BLOCK-END
