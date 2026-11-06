import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Try to find n=27 directly using efficient search
    # If that fails, fall back to n=26, then n=25, etc.
    
    # First, try to get a good n=26 solution quickly
    result_26 = efficient_search(26, time_limit=30.0)
    
    if result_26 is not None:
        # Try to extend to n=27
        result_27 = try_extend(result_26, time_limit=60.0)
        if result_27 is not None:
            return result_27
        return result_26
    
    # Fallback: try n=25
    result_25 = efficient_search(25, time_limit=30.0)
    if result_25 is not None:
        # Try to extend to n=26
        result_26b = try_extend(result_25, time_limit=60.0)
        if result_26b is not None:
            return result_26b
        return result_25
    
    # Absolute fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def efficient_search(n, time_limit=30.0):
    """Efficient local search with incremental violation updates."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute for each edge, which pairs of vertices (u,v) could form a C4 with it
    # Edge (i,j) is part of a C4 if there exist u,v such that (u,v) is an edge and
    # (u,i),(i,v),(v,j),(j,u) or similar pattern... Actually simpler:
    # A C4 is a set of 4 vertices {a,b,c,d} with edges (a,b),(b,c),(c,d),(d,a).
    # For a given edge (i,j), it participates in a C4 if there exist u,v with u≠v, u,v≠i,j
    # such that (i,u),(u,v),(v,j) are all edges of the same color, OR
    # (j,u),(u,v),(v,i) are all edges of the same color.
    
    # Better: count violations as number of C4's. Each C4 is counted once.
    # For efficiency, track for each color c, and each pair (u,v), the number of common neighbors in G_c.
    # A C4 exists iff some pair (u,v) has >= 2 common neighbors in G_c.
    
    # State: colors array, and for each color c, common_count[c][u][v] = number of common neighbors
    # of u and v in G_c.
    
    # Initialize with random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build initial common counts
    # adj_mask[c][v] = bitmask of neighbors of v in color c
    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)
    
    # common_count[c][u] is a list of bitmasks or we compute on the fly
    # For violation count: sum over c, over u<v, C(common_neighbors(c,u,v), 2)
    
    def count_violations(adj_mask):
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    current_violations = count_violations(adj_mask)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    best_violations = current_violations
    best_colors = colors[:]
    best_adj = [row[:] for row in adj_mask]
    
    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.9995
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            # Restart with new random solution
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj_mask = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            current_violations = count_violations(adj_mask)
            temperature = 2.0
            no_improve_count = 0
            continue
        
        # Pick a random edge to change
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try each possible new color
        best_new_color = old_color
        best_new_violations = current_violations
        
        for new_color in range(5):
            if new_color == old_color:
                continue
            
            # Tentatively change
            colors[e_idx] = new_color
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[new_color][i] |= (1 << j)
            adj_mask[new_color][j] |= (1 << i)
            
            new_violations = count_violations(adj_mask)
            
            if new_violations < best_new_violations:
                best_new_violations = new_violations
                best_new_color = new_color
            
            # Revert
            adj_mask[new_color][i] &= ~(1 << j)
            adj_mask[new_color][j] &= ~(1 << i)
            adj_mask[old_color][i] |= (1 << j)
            adj_mask[old_color][j] |= (1 << i)
            colors[e_idx] = old_color
        
        # Apply best new color if it improves
        if best_new_color != old_color:
            colors[e_idx] = best_new_color
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[best_new_color][i] |= (1 << j)
            adj_mask[best_new_color][j] |= (1 << i)
            
            delta = best_new_violations - current_violations
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_violations = best_new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    best_adj = [row[:] for row in adj_mask]
                    no_improve_count = 0
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improve_count += 1
            else:
                # Revert
                adj_mask[best_new_color][i] &= ~(1 << j)
                adj_mask[best_new_color][j] &= ~(1 << i)
                adj_mask[old_color][i] |= (1 << j)
                adj_mask[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05
    
    return None

def try_extend(base_result, time_limit=60.0):
    """Try to extend a valid coloring of K_n to K_{n+1}."""
    base_colors = base_result['colors']
    n_base = base_result['n']
    n_new = n_base + 1
    
    # Build adjacency masks for the base graph
    adj_mask = [[0] * n_base for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj_mask[c][i] |= (1 << j)
            adj_mask[c][j] |= (1 << i)
            idx += 1
    
    # new_colors[i] = color of edge (n_base, i) for i in 0..n_base-1
    # Violations: for each color c, for each pair (u,v) in neighbors of new vertex in color c,
    # check if u and v have a common neighbor in G_c (among the base vertices).
    
    def count_violations_new(new_colors):
        violations = 0
        for c in range(5):
            # Get neighbors of new vertex in color c
            neighbors = []
            for i in range(n_base):
                if new_colors[i] == c:
                    neighbors.append(i)
            
            # For each pair in neighbors, count common neighbors in G_c
            for a_idx in range(len(neighbors)):
                u = neighbors[a_idx]
                for b_idx in range(a_idx + 1, len(neighbors)):
                    v = neighbors[b_idx]
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 1:
                        violations += k  # Each common neighbor creates a C4
        return violations
    
    # Local search on new_colors
    best_new_colors = [random.randint(0, 4) for _ in range(n_base)]
    best_violations = count_violations_new(best_new_colors)
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        if best_violations == 0:
            break
        if no_improve > 3000:
            best_new_colors = [random.randint(0, 4) for _ in range(n_base)]
            best_violations = count_violations_new(best_new_colors)
            temperature = 1.0
            no_improve = 0
            continue
        
        e_idx = random.randint(0, n_base - 1)
        old_c = best_new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        
        best_new_colors[e_idx] = new_c
        new_violations = count_violations_new(best_new_colors)
        
        delta = new_violations - best_violations
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            best_violations = new_violations
            no_improve = 0
        else:
            best_new_colors[e_idx] = old_c
            no_improve += 1
        
        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05
    
    if best_violations == 0:
        # Build final coloring
        final_colors = [0] * (n_new * (n_new - 1) // 2)
        idx = 0
        for i in range(n_new):
            for j in range(i + 1, n_new):
                if j < n_base:
                    # Edge (i,j) in base graph
                    base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                    final_colors[idx] = base_colors[base_idx]
                else:
                    # j == n_base
                    final_colors[idx] = best_new_colors[i]
                idx += 1
        
        return {'n': n_new, 'colors': final_colors}
    
    return None
