import random
import time

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of unit distances.
    Uses a hexagonal spiral as a seed and improves it with local search.
    """
    # 1. Generate initial solution (Greedy Center-Out)
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
    
    # Sort by squared Euclidean distance from the origin
    candidates.sort(key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
    
    # Select the 32 closest points
    initial_points = set(candidates[:32])
    
    # Helper to count edges
    def count_edges(points):
        pts = list(points)
        edges = 0
        for i in range(len(pts)):
            a1, b1 = pts[i]
            for j in range(i + 1, len(pts)):
                a2, b2 = pts[j]
                da = a1 - a2
                db = b1 - b2
                # Distance squared is 1 if da^2 + da*db + db^2 == 1
                if da*da + da*db + db*db == 1:
                    edges += 1
        return edges
    
    best_points = initial_points
    best_score = count_edges(best_points)
    
    # 2. Local Search
    # Candidate pool for swaps: points with d^2 <= 25 (to allow some flexibility)
    pool = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            d2 = a*a + a*b + b*b
            if d2 <= 25:
                pool.append((a, b))
    
    start_time = time.time()
    deadline = 10.0 # 10 seconds for search
    
    current_points = initial_points
    current_score = best_score
    
    while time.time() - start_time < deadline:
        # Randomly pick a point to remove
        if not current_points:
            break
        p_remove = random.choice(list(current_points))
        
        # Randomly pick a point to add from pool (not in current)
        # To speed up, pick from pool not in current
        available = [p for p in pool if p not in current_points]
        if not available:
            break
        p_add = random.choice(available)
        
        new_points = set(current_points)
        new_points.remove(p_remove)
        new_points.add(p_add)
        
        new_score = count_edges(new_points)
        
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points
                
    return [[a, b] for a, b in best_points]
