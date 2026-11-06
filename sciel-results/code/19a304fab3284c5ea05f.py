import random
import math

def construct():
    n = 32
    
    # 1. Generate initial configuration: Hexagonal spiral
    # Points sorted by hex distance, then lexicographically
    # Hex distance d(q,r) = (|q| + |r| + |q+r|) / 2
    def hex_dist(q, r):
        return abs(q) + abs(r) + abs(q + r)
        
    # Generate a pool of candidate points
    # We need enough to pick 32 best
    # Max coordinate approx sqrt(32) ~ 6. Let's use -10 to 10.
    candidates = []
    for q in range(-10, 11):
        for r in range(-10, 11):
            candidates.append((q, r))
            
    # Sort by distance from origin
    candidates.sort(key=lambda p: (hex_dist(p[0], p[1]), p[0], p[1]))
    
    # Take the first n points
    points = set(candidates[:n])
    
    # 2. Local Search
    # Directions
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    def count_edges(point_set):
        edges = 0
        for p in point_set:
            for dq, dr in directions:
                if (p[0] + dq, p[1] + dr) in point_set:
                    edges += 1
        return edges // 2 # Each edge counted twice
    
    current_edges = count_edges(points)
    
    # Optimization loop
    # We try to swap a point in the set with a point not in the set
    # To keep it fast, we only consider points close to the current set
    max_iterations = 500
    for _ in range(max_iterations):
        improved = False
        # Create a list of points in the set
        point_list = list(points)
        random.shuffle(point_list)
        
        for p_in in point_list:
            # Generate candidate points to swap with
            # Look at neighbors of neighbors? Or just a random point in a bounding box?
            # Better: Look at points that are close to the set
            # We can generate a list of "external" candidates
            # For simplicity, let's try moving p_in to a nearby empty spot
            
            # Find empty neighbors of p_in? No, that just shifts.
            # We want to remove p_in and add p_out.
            
            # Let's define a candidate pool for p_out
            # It should be close to the set to maintain connectivity
            # We can pick from the next layer of the hexagon or random nearby
            
            # Simple heuristic: Try all 6 neighbors of p_in that are not in the set?
            # That's a "move".
            # Or try to swap p_in with a point at distance 1, 2, or 3.
            
            # Let's try a random point in a box around the set
            # Or better: Iterate through a pre-defined list of "potential" points
            
            # For efficiency, let's just try to move p_in to an empty neighbor
            # This is a "push" operation.
            
            for dq, dr in directions:
                new_p = (p_in[0] + dq, p_in[1] + dr)
                if new_p not in points:
                    # Try moving p_in to new_p
                    points.remove(p_in)
                    points.add(new_p)
                    new_edges = count_edges(points)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        improved = True
                        break
                    else:
                        points.remove(new_p)
                        points.add(p_in)
            
            if improved:
                break
            
            # Also try "swaps": remove p_in, add some other point
            # This is more expensive. Let's stick to moves for now.
            # If no move improves, we might be stuck.
            
        if not improved:
            # Try a more global search: Random swap
            # Pick a random point in set, pick a random point outside (but close)
            p_in = random.choice(list(points))
            # Generate a random point in a box of size 20
            p_out = (random.randint(-10, 10), random.randint(-10, 10))
            if p_out not in points:
                points.remove(p_in)
                points.add(p_out)
                new_edges = count_edges(points)
                if new_edges > current_edges:
                    current_edges = new_edges
                    improved = True
                else:
                    points.remove(p_out)
                    points.add(p_in)
            
            if not improved:
                break
                
    result = sorted([list(p) for p in points])
    return result
