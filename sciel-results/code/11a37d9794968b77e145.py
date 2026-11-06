import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N
DEADLINE = time.time() + 290  # Safety margin before 300s limit

def construct():
    start_time = time.time()
    
    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    # Precompute squared distances
    # dist2[i][j] = squared distance between point i and point j
    xs = np.array([p[0] for p in all_points], dtype=np.int32)
    ys = np.array([p[1] for p in all_points], dtype=np.int32)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)  # shape (GRID_SIZE, GRID_SIZE)
    
    # For fast validity checking:
    # A set S is valid if for every triple (a, b, c) in S, the triangle is NOT isosceles.
    # Triangle (a, b, c) is isosceles if:
    #   dist2[a][b] == dist2[a][c]  (apex at a)
    #   dist2[a][b] == dist2[b][c]  (apex at b)
    #   dist2[a][c] == dist2[b][c]  (apex at c)
    
    # Key insight: For a fixed set S, a point p can be added if:
    # 1. For every a in S, dist2[p][a] is unique among {dist2[p][b] : b in S} 
    #    (i.e., no two points in S are at the same distance from p)
    # 2. For every a in S, dist2[p][a] is not equal to dist2[a][b] for any b in S, b != a
    #    (i.e., no point a in S has p and some other point b in S at the same distance)
    
    # We'll use a greedy approach with random restarts, plus local improvement.
    
    best_set = []
    random.seed(42)
    
    def is_valid_set(indices):
        """Check if a set of indices forms a no-isosceles set."""
        m = len(indices)
        if m < 3:
            return True
        
        # Build set for fast lookup
        idx_set = set(indices)
        
        for i in range(m):
            a = indices[i]
            for j in range(i + 1, m):
                b = indices[j]
                dab2 = dist2[a][b]
                # Check if there exists c in S, c != a, c != b, such that:
                # dist2[a][c] == dab2 (apex at a) OR dist2[b][c] == dab2 (apex at b)
                # Also check if dist2[a][b] == dist2[a][c] or dist2[b][c] for any c
                
                # For efficiency, just check all triples
                for k in range(j + 1, m):
                    c = indices[k]
                    dac2 = dist2[a][c]
                    dbc2 = dist2[b][c]
                    if dab2 == dac2 or dab2 == dbc2 or dac2 == dbc2:
                        return False
        return True
    
    def can_add_point(p_idx, current_indices):
        """Check if p_idx can be added to the set defined by current_indices."""
        m = len(current_indices)
        if m < 2:
            return True
        
        for i in range(m):
            a = current_indices[i]
            dap2 = dist2[p_idx][a]
            
            for j in range(i + 1, m):
                b = current_indices[j]
                dab2 = dist2[a][b]
                dbp2 = dist2[p_idx][b]
                
                if dap2 == dab2 or dbp2 == dab2 or dap2 == dbp2:
                    return False
        return True
    
    # Greedy construction with multiple random orderings
    for attempt in range(500):
        if time.time() > DEADLINE:
            break
        
        # Random shuffle of points
        order = list(range(GRID_SIZE))
        random.shuffle(order)
        
        # Greedy: add points if compatible
        current_set = []
        
        for idx in order:
            if time.time() > DEADLINE:
                break
            
            if can_add_point(idx, current_set):
                current_set.append(idx)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # Local improvement: try to add more points
        current_pos_set = set(best_set)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for p_idx in range(GRID_SIZE):
                if p_idx in current_pos_set:
                    continue
                
                if can_add_point(p_idx, best_set):
                    best_set.append(p_idx)
                    current_pos_set.add(p_idx)
                    improved = True
        
        if len(best_set) >= 18:  # Good enough, stop early
            break
    
    # Convert indices to (x, y) tuples
    result = [all_points[i] for i in best_set]
    
    return result
