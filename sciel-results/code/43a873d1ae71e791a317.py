import time
import random

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # Build hex-2 core (19 points)
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a + b)) <= 2:
                hex2.add((a, b))
    
    # Build shell-3 (18 points)
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) == 3:
                shell3.append((a, b))
    
    # Greedily add 13 points from shell3
    current_set = set(hex2)
    selected_shell = set()
    
    for _ in range(13):
        best_point = None
        best_gain = -1
        for p in shell3:
            if p in selected_shell:
                continue
            gain = 0
            for q in current_set:
                if is_adjacent(p, q):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_point = p
        if best_point is not None:
            selected_shell.add(best_point)
            current_set.add(best_point)
    
    # Now we have 32 points. Compute initial score.
    pts = list(current_set)
    
    def count_edges(points):
        cnt = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    cnt += 1
        return cnt
    
    current_score = count_edges(pts)
    
    # Build a larger pool for hill climbing: hex-4
    pool = set()
    for a in range(-4, 5):
        for b in range(-4, 5):
            if max(abs(a), abs(b), abs(a + b)) <= 4:
                pool.add((a, b))
    
    # Hill climbing: try all single swaps (remove one, add one from pool)
    best_score = current_score
    best_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_swap_gain = 0
        best_remove = None
        best_add = None
        
        # Precompute neighbor counts for each point in current_set
        # deg_in[p] = number of neighbors of p in current_set
        deg_in = {}
        for p in current_set:
            deg_in[p] = 0
        for p in current_set:
            for q in current_set:
                if p != q and is_adjacent(p, q):
                    deg_in[p] += 1
        
        # For each candidate removal and addition, compute net gain
        for p_remove in current_set:
            # Edges lost when removing p_remove
            edges_lost = deg_in[p_remove]
            
            for p_add in pool:
                if p_add in current_set:
                    continue
                # Edges gained when adding p_add to (current_set - {p_remove})
                edges_gained = 0
                for q in current_set:
                    if q == p_remove:
                        continue
                    if is_adjacent(p_add, q):
                        edges_gained += 1
                # If p_add is adjacent to p_remove, that edge is also lost (but p_remove is removed)
                # Actually edges_gained already excludes p_remove, so it's correct.
                
                net_gain = edges_gained - edges_lost
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_remove = p_remove
                    best_add = p_add
        
        if best_remove is not None and best_add is not None:
            current_set.remove(best_remove)
            current_set.add(best_add)
            current_score += best_swap_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
    
    # Return the best set found
    result = [[a, b] for a, b in best_set]
    
    # Safety check: ensure exactly 32 distinct points within bounds
    if len(result) != 32:
        # Fallback: use the greedy result
        result = [[a, b] for a, b in current_set]
    
    # Verify all points are within bounds and distinct
    seen = set()
    valid = []
    for p in result:
        a, b = p
        if abs(a) <= 64 and abs(b) <= 64 and (a, b) not in seen:
            seen.add((a, b))
            valid.append([a, b])
    
    if len(valid) == 32:
        return valid
    
    # Last resort fallback: known valid 32-point configuration
    # Use hex-3 minus 5 carefully chosen points
    hex3 = set()
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) <= 3:
                hex3.add((a, b))
    
    # Remove 5 boundary points with lowest degree
    pts3 = list(hex3)
    for _ in range(5):
        min_deg = float('inf')
        min_idx = 0
        for i, p in enumerate(pts3):
            deg = 0
            for q in pts3:
                if p != q and is_adjacent(p, q):
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_idx = i
        pts3.pop(min_idx)
    
    return [[a, b] for a, b in pts3]
