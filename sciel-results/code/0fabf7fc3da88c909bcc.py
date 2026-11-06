import time
import random
import numpy as np
from collections import Counter

def _fast_score(points):
    """
    Calculate the maximum multiplicity of a positive doubled triangle area.
    points: list of 30 tuples (x, y)
    """
    P = np.array(points, dtype=np.int64)
    # Max possible doubled area for coords in [-30, 30] is |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Max diff is 60. Max term is 30*60 = 1800. Sum of 3 terms can be up to 5400.
    # Allocate a bit more to be safe.
    max_area = 6000
    counts = np.zeros(max_area, dtype=np.int64)
    
    n = len(P)
    for i in range(n):
        # Vectors from P[i] to all P[j] where j > i
        # This avoids double counting and self-intersections
        V = P[i+1:] - P[i]
        n_v = len(V)
        if n_v < 2:
            continue
        
        # We need cross products of all pairs (a, b) from V where a < b
        # V is shape (n_v, 2)
        # We can compute cross products for all pairs using broadcasting
        # V[:, 0] is shape (n_v, 1)
        # V[1:, 0] is shape (n_v-1, 1)
        # This is getting complicated to vectorize fully without huge memory spikes for n=30 (small enough though)
        # n_v is max 29. Pairs is ~400. 
        # Let's just use a small loop for the pairs within the vectorized pivot, 
        # but the original error was in the vectorization of vk.
        # Actually, for n=30, even pure python loops might be okay if optimized, 
        # but let's stick to the vectorized pivot idea correctly.
        
        # Correct vectorized cross product for all pairs in V:
        # v_i = V[i], v_j = V[j]
        # cross = v_i[0]*v_j[1] - v_i[1]*v_j[0]
        
        # We can create indices i < j
        idx_i, idx_j = np.triu_indices(n_v, k=1)
        vi = V[idx_i] # shape (num_pairs, 2)
        vj = V[idx_j] # shape (num_pairs, 2)
        
        cross = vi[:, 0] * vj[:, 1] - vi[:, 1] * vj[:, 0]
        areas = np.abs(cross)
        
        mask = areas > 0
        if np.any(mask):
            valid_areas = areas[mask]
            # Filter out any areas that exceed our array size (shouldn't happen with 6000)
            valid_areas = valid_areas[valid_areas < max_area]
            if len(valid_areas) > 0:
                bc = np.bincount(valid_areas, minlength=max_area)
                counts += bc
                
    return int(np.max(counts))

def _sa_search(initial_points, time_limit=5.0):
    """
    Simulated Annealing to improve the configuration.
    """
    start_time = time.time()
    points = [tuple(p) for p in initial_points]
    current_score = _fast_score(points)
    best_points = [p for p in points]
    best_score = current_score
    
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    
    occupied = set(points)
    
    # Pre-compute valid coordinates
    # We can't pre-compute all 3600 coords and filter easily in the loop for speed,
    # but we can pick a random point and a random new coord.
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = points[idx]
        
        # Pick a random new location
        # To ensure distinctness, we pick a random coord and check if it's occupied.
        # If it is, we reject the move (or pick another).
        # Given 30 points in 3600 spots, collision is rare (~0.8%).
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_p = (new_x, new_y)
        
        if new_p == old_p:
            continue
        if new_p in occupied:
            continue
            
        # Apply move
        points[idx] = new_p
        occupied.remove(old_p)
        occupied.add(new_p)
        
        new_score = _fast_score(points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p for p in points]
        else:
            # Reject move
            points[idx] = old_p
            occupied.remove(new_p)
            occupied.add(old_p)
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_points, best_score

def construct():
    start_time = time.time()
    hard_deadline = start_time + 58.0 # Leave 2s buffer for safety
    
    # Generate initial candidates
    candidates = []
    
    # 1. 5x6 Grid centered at 0
    g1 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    candidates.append(g1)
    
    # 2. 6x5 Grid
    g2 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]
    candidates.append(g2)
    
    # 3. 3x10 Grid
    g3 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
    candidates.append(g3)
    
    # 4. 10x3 Grid
    g4 = [(x, y) for x in range(-5, 5) for y in range(-1, 2)]
    candidates.append(g4)
    
    # 5. Random points
    random.seed(123)
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    g5 = random.sample(all_coords, 30)
    candidates.append(g5)
    
    # 6. Perturbed Grid (5x6)
    g6_base = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
    g6 = []
    for p in g6_base:
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        new_p = (p[0] + dx, p[1] + dy)
        if -30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30:
            g6.append(new_p)
        else:
            g6.append(p)
    if len(set(g6)) != 30:
        g6 = list(g1)
    candidates.append(g6)
    
    # Add a few more structured candidates: Sheared grids
    # Shear the 5x6 grid
    g7 = [(x, y + x) for x, y in g1]
    # Check bounds and distinctness
    valid_g7 = [p for p in g7 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid_g7)) == 30:
        candidates.append(valid_g7)

    # Shear the 6x5 grid
    g8 = [(x + y, y) for x, y in g2]
    valid_g8 = [p for p in g8 if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    if len(set(valid_g8)) == 30:
        candidates.append(valid_g8)

    best_points = None
    best_score = -1
    
    # Evaluate initial candidates
    scored_candidates = []
    for c in candidates:
        if len(c) != 30 or len(set(c)) != 30:
            continue
        s = _fast_score(c)
        scored_candidates.append((s, c))
        if s > best_score:
            best_score = s
            best_points = [tuple(p) for p in c]
            
    if not scored_candidates:
        return [tuple(p) for p in g1]

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Run SA on top candidates
    # Allocate time dynamically
    remaining = hard_deadline - time.time() - 2.0
    
    # We will run SA on the top 3 distinct candidates
    used_seeds = set()
    for i in range(min(3, len(scored_candidates))):
        if time.time() > hard_deadline - 2.0:
            break
            
        s, c = scored_candidates[i]
        # Avoid running SA on identical candidates
        key = tuple(sorted(c))
        if key in used_seeds:
            continue
        used_seeds.add(key)
        
        time_left = hard_deadline - time.time() - 2.0
        if time_left < 2.0:
            break
            
        # Allocate time: first gets 40% of remaining, second 30%, third 30%
        if i == 0:
            t_limit = min(0.4 * remaining, time_left)
        elif i == 1:
            t_limit = min(0.3 * remaining, time_left)
        else:
            t_limit = min(0.3 * remaining, time_left)
            
        t_limit = max(1.0, min(t_limit, time_left))
        
        fp, fs = _sa_search(c, time_limit=t_limit)
        if fs > best_score:
            best_score = fs
            best_points = fp
                
    if best_points is None:
        best_points = [tuple(p) for p in scored_candidates[0][1]]
        
    return [tuple(p) for p in best_points]

# Quick test to ensure no errors
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_fast_score(pts)}")
    print(f"Points: {pts}")
