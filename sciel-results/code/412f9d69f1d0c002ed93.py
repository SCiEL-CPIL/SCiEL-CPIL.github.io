"""
Five-colourings of complete graphs without monochromatic C4 (R_5(C4)).

Uses a fast bitset-based local search to find a valid 5-coloring of K_n.
Targets n=26 (known Lazebnik-Woldar bound) and attempts n=27.
"""

import time
import random
import numpy as np

_DEADLINE = 145.0

def _local_search_bitset(n, time_budget):
    """
    Local search using bitmasks for speed.
    neighbors[c][u] is an integer where bit v is 1 if edge (u,v) has color c.
    """
    start_time = time.time()
    random.seed(42)
    
    # Initialize with random coloring
    # colors[i, j] is the color of edge (i, j)
    colors = np.random.randint(0, 5, (n, n))
    np.fill_diagonal(colors, -1)
    colors = np.triu(colors, 1) + np.triu(colors, 1).T
    
    # Build initial adjacency bitmasks
    # neighbors[c][u] is a bitmask of neighbors of u in color c
    neighbors = [[0] * n for _ in range(5)]
    
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[i, j]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if u and v have >= 2 common neighbors in color c
                    # We only count violations if edge (u,v) is NOT color c? 
                    # No, the condition is: No colour class may contain a four-cycle.
                    # A C4 in color c consists of vertices u, v, w, z such that
                    # (u,v), (v,w), (w,z), (z,u) are all color c.
                    # This is equivalent to u and v having two common neighbors in color c.
                    # Wait, the definition says "No two vertices with two common neighbours in one colour".
                    # This applies to ANY pair of vertices u, v, regardless of the color of edge (u,v).
                    # If u and v have two common neighbors in color c, then those two neighbors
                    # plus u and v form a C4 in the subgraph of color c?
                    # Let neighbors be a, b.
                    # Edges (u,a), (v,a), (u,b), (v,b) are all color c.
                    # This forms a K_{2,2} in the color c subgraph, which contains a C4.
                    # Yes. So we check all pairs (u, v).
                    common = neighbors[c][u] & neighbors[c][v]
                    if common.bit_count() >= 2:
                        total += 1
        return total

    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors.copy()
    
    # Precompute edge list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        if current_violations == 0:
            break
            
        # Pick a random edge
        u, v = edges[random.randint(0, len(edges) - 1)]
        old_c = colors[u, v]
        
        best_new_c = old_c
        best_local_violations = current_violations
        
        # Try all 5 colors
        for new_c in range(5):
            if new_c == old_c:
                continue
                
            # Update neighbors
            neighbors[old_c][u] &= ~(1 << v)
            neighbors[old_c][v] &= ~(1 << u)
            neighbors[new_c][u] |= (1 << v)
            neighbors[new_c][v] |= (1 << u)
            
            # Count violations
            # Optimization: Only check violations involving u or v?
            # A change in edge (u,v) affects:
            # 1. Pairs (u, x) for all x != v. Common neighbors of u and x in color c might change.
            # 2. Pairs (v, x) for all x != u.
            # 3. The pair (u, v) itself.
            # However, implementing incremental violation counting is complex.
            # With bitsets, full count is O(5 * n^2 * bit_count).
            # For n=26, n^2 = 676. 5 * 676 = 3380 checks.
            # bit_count is very fast.
            # Let's just do full count first. If it's too slow, we optimize.
            # 3380 checks per move. If we do 100k moves, that's 338M checks.
            # Python is slow. 338M operations might take ~30-60s.
            # We need to be smarter or accept slower speed.
            # Actually, `common.bit_count()` is a C-level operation.
            # The loop overhead is the bottleneck.
            
            new_violations = count_violations()
            
            if new_violations < best_local_violations:
                best_local_violations = new_violations
                best_new_c = new_c
            else:
                # Revert
                neighbors[new_c][u] &= ~(1 << v)
                neighbors[new_c][v] &= ~(1 << u)
                neighbors[old_c][u] |= (1 << v)
                neighbors[old_c][v] |= (1 << u)
        
        # Apply best move
        if best_new_c != old_c:
            colors[u, v] = best_new_c
            colors[v, u] = best_new_c
            neighbors[old_c][u] &= ~(1 << v)
            neighbors[old_c][v] &= ~(1 << u)
            neighbors[best_new_c][u] |= (1 << v)
            neighbors[best_new_c][v] |= (1 << u)
            current_violations = best_local_violations
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Random restart or kick
            if no_improve_count > 50:
                for _ in range(5):
                    u2, v2 = edges[random.randint(0, len(edges) - 1)]
                    old_c2 = colors[u2, v2]
                    new_c2 = random.randint(0, 4)
                    colors[u2, v2] = new_c2
                    colors[v2, u2] = new_c2
                    neighbors[old_c2][u2] &= ~(1 << v2)
                    neighbors[old_c2][v2] &= ~(1 << u2)
                    neighbors[new_c2][u2] |= (1 << v2)
                    neighbors[new_c2][v2] |= (1 << u2)
                current_violations = count_violations()
                no_improve_count = 0
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
    return best_colors, best_violations

def construct():
    start_time = time.time()
    
    # Target n=26
    n_target = 26
    # Leave 5s for output and safety
    time_budget = _DEADLINE - (time.time() - start_time) - 5.0
    
    colors, violations = _local_search_bitset(n_target, time_budget)
    
    if violations == 0:
        # Extract colors in lexicographic order
        res_colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                res_colors.append(int(colors[i, j]))
        
        # Check if we have time to try n=27
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 20.0: # At least 20s left
             n_target_27 = 27
             time_budget_27 = _DEADLINE - (time.time() - start_time) - 5.0
             colors_27, violations_27 = _local_search_bitset(n_target_27, time_budget_27)
             if violations_27 == 0:
                 res_colors_27 = []
                 for i in range(n_target_27):
                     for j in range(i + 1, n_target_27):
                         res_colors_27.append(int(colors_27[i, j]))
                 return {'n': n_target_27, 'colors': res_colors_27}
        
        return {'n': n_target, 'colors': res_colors}
    
    # If n=26 failed, try n=25
    if time.time() - start_time < _DEADLINE - 10.0:
        n_target = 25
        time_budget = _DEADLINE - (time.time() - start_time) - 5.0
        colors, violations = _local_search_bitset(n_target, time_budget)
        if violations == 0:
            res_colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    res_colors.append(int(colors[i, j]))
            return {'n': n_target, 'colors': res_colors}
            
    # Final fallback: n=11
    n = 11
    colors_list = []
    for i in range(n):
        for j in range(i + 1, n):
            # Simple cyclic coloring
            colors_list.append((j - i) % 5)
    return {'n': n, 'colors': colors_list}
