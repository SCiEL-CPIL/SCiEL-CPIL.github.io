import math

def construct():
    n = 32
    # Generate candidate points in a range large enough to cover 32 points
    # The hexagon of radius 3 has 37 points, so radius 3 is sufficient.
    # Let's use radius 4 to be safe.
    R = 4
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            # Check if the point is within the hexagonal distance R
            # Hex distance in axial coordinates: (|a| + |b| + |a+b|) / 2
            hex_dist = (abs(a) + abs(b) + abs(a + b)) // 2
            if hex_dist <= R:
                points.append((a, b))
                
    # Sort by hex distance, then by some tie-breaker to ensure determinism
    # We want the most compact shape. Sorting by hex distance is good.
    # Tie-breaking: we can sort by angle or just lexicographically.
    # Lexicographical might not be optimal for compactness if it biases direction,
    # but for a hexagon, any subset of the outer ring that is contiguous will work.
    # Actually, simply taking the first N points from a sorted list by distance
    # might not guarantee the *maximum* edges if the tie-breaking is bad.
    # However, for n=32, the optimal shape is well-known.
    
    # A more robust method: 
    # 1. Start with the center (0,0).
    # 2. Add points from the outer rings one by one, choosing the point that 
    #    maximizes the number of new edges formed with already selected points.
    # This is a greedy algorithm that is very likely to find the optimal 76 edges.
    
    # Let's implement the greedy approach.
    # Pre-calculate neighbors for a set of candidate points.
    
    # Candidate pool: all points within hex distance 3 (since 32 < 37)
    # Actually, we might need points from radius 3.
    # Let's use a pool of points with hex distance <= 3.
    pool = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            hex_dist = (abs(a) + abs(b) + abs(a + b)) // 2
            if hex_dist <= 3:
                pool.append((a, b))
                
    # Map point to index for fast lookup
    point_to_idx = {p: i for i, p in enumerate(pool)}
    
    # Pre-calculate adjacency within the pool
    # Two points are adjacent if their squared distance is 1.
    # d^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    def is_adjacent(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + db*db + da*db == 1
        
    # Build adjacency list for the pool
    adj = {p: [] for p in pool}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if is_adjacent(pool[i], pool[j]):
                adj[pool[i]].append(pool[j])
                adj[pool[j]].append(pool[i])
                
    # Greedy selection
    selected = [(0, 0)]
    selected_set = {(0, 0)}
    
    while len(selected) < n:
        best_point = None
        best_edges = -1
        
        for p in pool:
            if p in selected_set:
                continue
            
            # Count how many neighbors of p are already in selected
            new_edges = 0
            for neighbor in adj[p]:
                if neighbor in selected_set:
                    new_edges += 1
            
            if new_edges > best_edges:
                best_edges = new_edges
                best_point = p
                
        if best_point is None:
            break
            
        selected.append(best_point)
        selected_set.add(best_point)
        
    return [[a, b] for a, b in selected]
