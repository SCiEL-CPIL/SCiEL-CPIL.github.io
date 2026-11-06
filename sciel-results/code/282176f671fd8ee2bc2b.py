import time
import random

def _neighbors(a, b):
    return [
        (a + 1, b), (a - 1, b),
        (a, b + 1), (a, b - 1),
        (a + 1, b - 1), (a - 1, b + 1)
    ]

def _dist_sq(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db

def _edge_count(S):
    """Count pairs in set S at distance exactly 1."""
    S_set = set(S)
    count = 0
    for p in S_set:
        for q in _neighbors(p[0], p[1]):
            if q in S_set and p < q:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s budget, leaving 50s safety margin
    
    # 1. Generate candidate points from a large enough region
    # r=3 hexagon has 37 points. r=4 has 61 points.
    # We want 32 points.
    # Let's start with the r=3 hexagon (37 points) and greedily remove 5.
    
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a - b) <= 3:
                points.append((a, b))
                
    # Greedy removal to get 32 points
    current_set = set(points)
    while len(current_set) > 32:
        min_deg = float('inf')
        point_to_remove = None
        for p in current_set:
            deg = 0
            for q in _neighbors(p[0], p[1]):
                if q in current_set:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                point_to_remove = p
        current_set.remove(point_to_remove)
        
    best_set = current_set
    best_edges = _edge_count(best_set)
    
    # 2. Local Search / Simulated Annealing
    # We will try to improve the edge count.
    # Strategy: Randomly pick a point, move it to a random nearby empty spot.
    # If edges increase, accept. If not, accept with probability exp(-delta/T) (annealing).
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Define a search space for moves.
    # Points can move within a larger bounding box, say r=6.
    # But we should stay close to the current configuration.
    
    while time.time() < deadline:
        if len(best_set) != 32:
            break
            
        # Pick a random point from the set
        p = random.choice(list(best_set))
        
        # Generate a random neighbor position for p
        # We look for an empty spot within distance 2 or 3
        new_pos = None
        for _ in range(20):
            da = random.randint(-2, 2)
            db = random.randint(-2, 2)
            if da == 0 and db == 0:
                continue
            candidate = (p[0] + da, p[1] + db)
            if candidate not in best_set:
                new_pos = candidate
                break
                
        if new_pos is None:
            continue
            
        # Calculate edge count change
        # Remove p, add new_pos
        # Old edges involving p
        old_edges_p = 0
        for q in _neighbors(p[0], p[1]):
            if q in best_set and q != p:
                old_edges_p += 1
                
        # New edges involving new_pos
        new_edges_np = 0
        for q in _neighbors(new_pos[0], new_pos[1]):
            if q in best_set and q != p and q != new_pos:
                new_edges_np += 1
                
        delta = new_edges_np - old_edges_p
        
        if delta >= 0 or random.random() < pow(2.71828, -abs(delta) / T):
            # Accept move
            best_set.remove(p)
            best_set.add(new_pos)
            
            # Update best if improved
            if delta > 0:
                best_edges += delta
                # Reset temperature or keep it? Keep it for now.
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Convert to list of lists
    return [list(p) for p in best_set]
