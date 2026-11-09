"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Precompute all 60 distinct 6-cycles on 6 labeled vertices
    _cycles_cache = None
    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            for perm in permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[i], cycle[(i+1)%6])) for i in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycles.append(cycle)
            _cycles_cache = cycles
        return _cycles_cache
    
    def _has_c6_in_subset(adj_matrix, subset, color):
        """Check if induced subgraph on subset in color class contains a C6."""
        n_sub = len(subset)
        if n_sub < 6:
            return False
        cycles = _get_cycles()
        for cycle in cycles:
            ok = True
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                if adj_matrix[u][w] != color:
                    ok = False
                    break
            if ok:
                return True
        return False
    
    def _verify_full(colors, n):
        """Full verification that no color class contains a C6."""
        if n < 6:
            return True
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Build adjacency lists per color
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        for c in range(4):
            # Check all 6-subsets
            for subset in combinations(range(n), 6):
                if _has_c6_in_subset(adj, subset, c):
                    return False
        return True
    
    def _local_search_fix(colors, n, deadline, max_iters=5000):
        """Local search to fix violations by flipping edges."""
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Check if valid
            if _verify_full(colors, n):
                return colors, True
            
            # Find a violating C6
            found_violation = False
            for c in range(4):
                for subset in combinations(range(n), 6):
                    if _has_c6_in_subset(adj, subset, c):
                        # Found a C6, try to fix by flipping one edge
                        cycles = _get_cycles()
                        for cycle in cycles:
                            ok = True
                            for k in range(6):
                                u = subset[cycle[k]]
                                w = subset[cycle[(k+1)%6]]
                                if adj[u][w] != c:
                                    ok = False
                                    break
                            if ok:
                                # Flip a random edge in the cycle
                                k = np.random.randint(6)
                                u = subset[cycle[k]]
                                w = subset[cycle[(k+1)%6]]
                                if u > w:
                                    u, w = w, u
                                # Find index
                                idx = 0
                                for i in range(n):
                                    for j in range(i+1, n):
                                        if i == u and j == w:
                                            break
                                        idx += 1
                                    else:
                                        continue
                                    break
                                
                                old_c = colors[idx]
                                new_c = np.random.randint(0, 4)
                                while new_c == old_c:
                                    new_c = np.random.randint(0, 4)
                                
                                # Update structures
                                adj_list[old_c][u].discard(w)
                                adj_list[old_c][w].discard(u)
                                adj[u][w] = new_c
                                adj[w][u] = new_c
                                adj_list[new_c][u].add(w)
                                adj_list[new_c][w].add(u)
                                colors[idx] = new_c
                                found_violation = True
                                break
                        if found_violation:
                            break
                if found_violation:
                    break
            
            if not found_violation:
                break
        
        return colors, _verify_full(colors, n)
    
    def _backtrack_construct(n_target, deadline):
        """Incremental backtracking to construct a valid coloring."""
        adj = [[-1]*n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        colors = [-1] * (n_target*(n_target-1)//2)
        
        def _edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i*(n_target-1) - i*(i-1)//2 + (j - i - 1)
        
        def _check_c6_incremental(v, i, c):
            """Check if adding edge (i,v) with color c creates a C6 in color c."""
            # A C6 involving edge (i,v) in color c would be:
            # v - i - a - b - e - d - v where all edges are color c
            # So we need a path of length 4 from i to some neighbor d of v in color c
            # where d != i, and all intermediate vertices are distinct.
            # More precisely: for any neighbor d of v in color c (d != i),
            # check if there's a simple path i-a-b-e-d in G_c - {v} 
            # where a,b,e are distinct and not equal to i,d.
            
            neighbors_of_v_in_c = adj_list[c][v]
            for d in neighbors_of_v_in_c:
                if d == i:
                    continue
                # Check if there's a path of length 4 from i to d in G_c without using v
                # Path: i - a - b - e - d
                # a is neighbor of i in color c, a != v, a != d
                # b is neighbor of a in color c, b != v, b != i, b != d, b != a
                # e is neighbor of b in color c, e != v, e != i, e != d, e != a, e != b
                # and e is connected to d in color c
                found_path = False
                for a in adj_list[c][i]:
                    if a == v or a == d:
                        continue
                    for b in adj_list[c][a]:
                        if b == v or b == i or b == d or b == a:
                            continue
                        for e in adj_list[c][b]:
                            if e == v or e == i or e == d or e == a or e == b:
                                continue
                            if d in adj_list[c][e]:
                                found_path = True
                                break
                        if found_path:
                            break
                    if found_path:
                        break
                if found_path:
                    return False
            return True
        
        def _search(v):
            if time.time() > deadline:
                return False
            
            if v == n_target:
                return True
            
            if v == 0:
                return _search(v + 1)
            
            # Assign colors for edges (i, v) for i = 0, 1, ..., v-1
            def _assign(i):
                if time.time() > deadline:
                    return False
                
                if i == v:
                    return _search(v + 1)
                
                # Try each color
                for c in range(4):
                    if _check_c6_incremental(v, i, c):
                        adj[i][v] = c
                        adj[v][i] = c
                        adj_list[c][i].add(v)
                        adj_list[c][v].add(i)
                        colors[_edge_idx(i, v)] = c
                        
                        if _assign(i + 1):
                            return True
                        
                        # Backtrack
                        adj[i][v] = -1
                        adj[v][i] = -1
                        adj_list[c][i].discard(v)
                        adj_list[c][v].discard(i)
                        colors[_edge_idx(i, v)] = -1
                
                return False
            
            return _assign(0)
        
        if _search(0):
            return colors
        return None
    
    # Start with small n and try to grow
    best_n = 5
    best_colors = [0] * (best_n*(best_n-1)//2)
    
    # Try to construct valid colorings for increasing n
    for target_n in range(6, 21):
        if time.time() > deadline - 10:
            break
        
        success = False
        # Try backtracking first
        result = _backtrack_construct(target_n, deadline - 5)
        if result is not None:
            if _verify_full(result, target_n):
                best_n = target_n
                best_colors = result
                success = True
        
        if not success:
            # Try random + local search as fallback
            m = target_n * (target_n - 1) // 2
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                colors = np.random.randint(0, 4, size=m).tolist()
                colors, valid = _local_search_fix(colors, target_n, deadline - 5, max_iters=2000)
                if valid:
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break
        
        if success:
            continue  # Try larger n
        else:
            # If we can't find for target_n, try target_n - 1 as last resort
            if target_n > 6:
                target_n -= 1
                result = _backtrack_construct(target_n, deadline - 5)
                if result is not None and _verify_full(result, target_n):
                    best_n = target_n
                    best_colors = result
            break
    
    # Final safety: ensure we have a valid result
    if best_n >= 6 and not _verify_full(best_colors, best_n):
        # Fallback to a simple valid coloring
        best_n = 5
        best_colors = [0] * (best_n*(best_n-1)//2)
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
