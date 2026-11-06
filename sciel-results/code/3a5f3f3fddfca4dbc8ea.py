import numpy as np
import time
import random

def _score(points):
    """
    Calculate the maximum multiplicity of a positive doubled area.
    points: np.array of shape (30, 2) with integer coordinates.
    Returns: (max_count, max_area_value)
    """
    if points.size == 0:
        return 0, 0
    
    n = len(points)
    # Compute doubled areas for all triples
    # To be fast, we can compute pairwise differences and then cross products,
    # but for n=30, O(n^3) with numpy vectorization per pair or just a tight loop is fine.
    # Let's try a vectorized approach for speed.
    
    # points is (30, 2)
    # We want area2 for i<j<k.
    # area2 = |(p_j - p_i) x (p_k - p_i)|
    
    # A fully vectorized O(n^3) might use too much memory if not careful, 
    # but 30^3 = 27000, which is tiny.
    # However, creating a (30, 30, 30) array is wasteful.
    
    # Let's stick to a loop over i, but vectorize j,k?
    # Or just a tight Python loop since 4060 iterations is very fast in Python too.
    
    counts = {}
    max_count = 0
    best_area = 0
    
    # Precompute? No, just loop.
    # Using local variables for speed
    p = points
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, n):
            pj = p[j]
            # Vectorized cross product for all k > j?
            # v1 = pj - pi
            # v2 = p[k] - pi
            # cross = v1[0]*v2[1] - v1[1]*v2[0]
            
            dx1 = pj[0] - pi[0]
            dy1 = pj[1] - pi[1]
            
            # Get all k > j
            # We need abs(dx1 * (p[k,1] - pi[1]) - dy1 * (p[k,0] - pi[0]))
            # Let's just loop k, it's only 30*30*30/6 ~ 4500 ops.
            for k in range(j + 1, n):
                pk = p[k]
                dx2 = pk[0] - pi[0]
                dy2 = pk[1] - pi[1]
                area2 = abs(dx1 * dy2 - dy1 * dx2)
                if area2 > 0:
                    counts[area2] = counts.get(area2, 0) + 1
    
    if not counts:
        return 0, 0
        
    # Find max
    for area, count in counts.items():
        if count > max_count:
            max_count = count
            best_area = area
            
    return max_count, best_area

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial configuration: Random points in [-30, 30]^2
    # Ensure distinctness
    points = set()
    while len(points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        points.add((x, y))
    
    current_points = np.array(list(points), dtype=np.int64)
    current_count, current_area = _score(current_points)
    
    best_points = current_points.copy()
    best_count = current_count
    best_area = current_area
    
    # Local Search: Move one point at a time
    # We want to maximize the count of the MOST FREQUENT area.
    # This is tricky because moving a point changes the areas of all triples involving it.
    # Strategy: Pick a random point, move it to a random new location.
    # If the new score (max multiplicity) is better, accept.
    # If slightly worse, accept with probability exp(-delta/T) (Simulated Annealing).
    
    # However, _score returns the max over ALL areas.
    # We might want to bias towards increasing the count of the CURRENT best area.
    # Let's try a simple hill climbing first, then annealing if time permits.
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random index to move
        idx = random.randint(0, 29)
        old_pos = current_points[idx].copy()
        
        # Generate new position
        # Bias towards staying close? Or random jump?
        # Random jump in [-30, 30]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        
        # Check distinctness
        # Since we only move one point, we just need to ensure (new_x, new_y) != current_points[j] for j != idx
        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Try the move
        current_points[idx] = [new_x, new_y]
        new_count, new_area = _score(current_points)
        
        # Evaluate
        # We care about the global max multiplicity.
        delta = new_count - current_count
        
        if delta >= 0:
            # Better or equal, accept
            current_count = new_count
            current_area = new_area
            
            if new_count > best_count:
                best_count = new_count
                best_area = new_area
                best_points = current_points.copy()
        else:
            # Worse
            # Accept with probability exp(delta / T)
            if random.random() < np.exp(delta / T):
                current_count = new_count
                current_area = new_area
            else:
                # Revert
                current_points[idx] = old_pos
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_points.tolist()

# For testing/debugging, but construct is the entry point
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Score:", _score(np.array(pts)))
