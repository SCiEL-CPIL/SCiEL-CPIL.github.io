import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Try n=26, 27, 28
    # The known bound is 26. 27/28 are very hard.
    # We will focus on finding a valid 26 quickly.
    
    for n in [26, 27, 28]:
        if time.time() > deadline - 15:
            break
        result = min_conflict_search(n, time_limit=40)
        if result is not None:
            return result
            
    # Fallback to smaller n if 26 fails
    for n in [25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 5:
            break
        result = min_conflict_search(n, time_limit=5)
        if result is not None:
            return result
            
    # Final fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def min_conflict_search(n, time_limit=40):
    """
    Min-Conflict Heuristic for 5-coloring K_n with no monochromatic C4.
    """
    start_time = time.time()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Precompute edge indices for O(1) access
    edge_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_map[(i, j)] = idx
        edge_map[(j, i)] = idx

    def count_violations(neighbors):
        total = 0
        for c in range(5):
            for u in range(n):
                mask_u = neighbors[c][u]
                for v in range(u + 1, n):
                    common = mask_u & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        total += cnt * (cnt - 1) // 2
        return total

    best_result = None
    
    # Try a few different initializations
    for init_strategy in range(5):
        if time.time() > start_time + time_limit:
            break
            
        # Reset
        neighbors = [[0] * n for _ in range(5)]
        colors = [0] * num_edges
        
        if init_strategy == 0:
            # Random
            for idx in range(num_edges):
                c = random.randint(0, 4)
                colors[idx] = c
                i, j = edges[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
        else:
            # Cyclic-like: color based on difference
            for idx, (i, j) in enumerate(edges):
                diff = (j - i) % n
                c = diff % 5
                colors[idx] = c
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                
        current_violations = count_violations(neighbors)
        
        if current_violations == 0:
            return {'n': n, 'colors': colors}
            
        # Min-Conflict Loop
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < start_time + time_limit:
            if current_violations == 0:
                return {'n': n, 'colors': colors}
                
            move_made = False
            # Try to find a move that reduces violations
            for _ in range(20): # Try a few random probes
                u = random.randint(0, n - 1)
                # Check each color
                for c in random.sample(range(5), 5):
                    mask_u = neighbors[c][u]
                    if mask_u.bit_count() < 2:
                        continue
                    # Get list of neighbors
                    neigh_list = []
                    m = mask_u
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        neigh_list.append(idx)
                        m ^= lsb
                    if len(neigh_list) < 2:
                        continue
                    v, w = random.sample(neigh_list, 2)
                    # Check if v and w have a common neighbor in color c (other than u)
                    common_vw = neighbors[c][v] & neighbors[c][w]
                    common_vw &= ~(1 << u)
                    if common_vw.bit_count() > 0:
                        # Found a C4: u, v, x, w (where x is a common neighbor)
                        # Recolor edge (u, v)
                        idx_uv = edge_map[(u, v)]
                        old_c = colors[idx_uv]
                        new_c = random.randint(0, 4)
                        while new_c == old_c:
                            new_c = random.randint(0, 4)
                        
                        # Apply move
                        i, j = edges[idx_uv]
                        neighbors[old_c][i] &= ~(1 << j)
                        neighbors[old_c][j] &= ~(1 << i)
                        neighbors[new_c][i] |= (1 << j)
                        neighbors[new_c][j] |= (1 << i)
                        colors[idx_uv] = new_c
                        
                        new_violations = count_violations(neighbors)
                        
                        if new_violations < current_violations:
                            current_violations = new_violations
                            no_improve_count = 0
                            move_made = True
                            break
                        else:
                            # Revert
                            neighbors[new_c][i] &= ~(1 << j)
                            neighbors[new_c][j] &= ~(1 << i)
                            neighbors[old_c][i] |= (1 << j)
                            neighbors[old_c][j] |= (1 << i)
                            colors[idx_uv] = old_c
                            
            if not move_made:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Accept a random move to escape local minimum
                    idx = random.randint(0, num_edges - 1)
                    old_c = colors[idx]
                    new_c = random.randint(0, 4)
                    while new_c == old_c:
                        new_c = random.randint(0, 4)
                    
                    i, j = edges[idx]
                    neighbors[old_c][i] &= ~(1 << j)
                    neighbors[old_c][j] &= ~(1 << i)
                    neighbors[new_c][i] |= (1 << j)
                    neighbors[new_c][j] |= (1 << i)
                    colors[idx] = new_c
                    
                    current_violations = count_violations(neighbors)
                    no_improve_count = 0

    return None
