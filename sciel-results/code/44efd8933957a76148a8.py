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

    def _edge_index(i, j, n):
        """Compute index for edge (i,j) in lexicographic order."""
        if i > j:
            i, j = j, i
        return i*(n-1) - i*(i-1)//2 + (j - i - 1)

    def _build_adj_structures(colors, n):
        """Build adjacency matrix and per-color adjacency lists."""
        adj = [[-1]*n for _ in range(n)]
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
                idx += 1
        return adj, adj_list

    def _find_c6(adj, adj_list, n):
        """Find a C6 in any color class. Returns (color, subset, cycle) or None."""
        cycles = _get_cycles()
        for c in range(4):
            # For each color, check if any 6-subset contains a C6
            for subset in combinations(range(n), 6):
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k+1)%6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return c, subset, cycle
        return None

    def _verify_full(colors, n):
        """Full verification that no color class contains a C6."""
        if n < 6:
            return True
        adj, adj_list = _build_adj_structures(colors, n)
        return _find_c6(adj, adj_list, n) is None

    def _local_search_fix(colors, n, deadline, max_iters=10000):
        """Local search to fix violations by flipping edges."""
        adj, adj_list = _build_adj_structures(colors, n)
        cycles = _get_cycles()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find a violating C6
            result = _find_c6(adj, adj_list, n)
            if result is None:
                return colors, True
            
            c, subset, cycle = result
            
            # Try to fix by flipping one edge in the cycle
            # Try all 6 edges, pick the one that minimizes new violations
            best_flip = None
            best_violation_count = float('inf')
            
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                if u > w:
                    u, w = w, u
                
                old_c = adj[u][w]
                
                # Try each new color
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    
                    # Check if this flip would create a new C6
                    # Temporarily flip and check
                    adj_list[old_c][u].discard(w)
                    adj_list[old_c][w].discard(u)
                    adj[u][w] = new_c
                    adj[w][u] = new_c
                    adj_list[new_c][u].add(w)
                    adj_list[new_c][w].add(u)
                    
                    # Count violations (approximate: just check if still has C6)
                    # For efficiency, just check if we fixed the current one
                    still_has_current = False
                    for ck in range(6):
                        uu = subset[cycle[ck]]
                        ww = subset[cycle[(ck+1)%6]]
                        if adj[uu][ww] != c:
                            still_has_current = True
                            break
                    
                    if still_has_current:
                        # This flip didn't fix the current violation
                        # Undo and try next
                        adj_list[new_c][u].discard(w)
                        adj_list[new_c][w].discard(u)
                        adj[u][w] = old_c
                        adj[w][u] = old_c
                        adj_list[old_c][u].add(w)
                        adj_list[old_c][w].add(u)
                        continue
                    
                    # This flip fixed the current C6. Accept it.
                    idx = _edge_index(u, w, n)
                    colors[idx] = new_c
                    best_flip = (u, w, old_c, new_c)
                    break
                
                if best_flip:
                    break
                
                # Undo if we didn't break
                if adj[u][w] != old_c:
                    adj_list[new_c][u].discard(w)
                    adj_list[new_c][w].discard(u)
                    adj[u][w] = old_c
                    adj[w][u] = old_c
                    adj_list[old_c][u].add(w)
                    adj_list[old_c][w].add(u)
            
            if best_flip is None:
                # No good flip found, do a random flip
                k = np.random.randint(6)
                u = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                if u > w:
                    u, w = w, u
                old_c = adj[u][w]
                new_c = np.random.randint(0, 4)
                while new_c == old_c:
                    new_c = np.random.randint(0, 4)
                
                adj_list[old_c][u].discard(w)
                adj_list[old_c][w].discard(u)
                adj[u][w] = new_c
                adj[w][u] = new_c
                adj_list[new_c][u].add(w)
                adj_list[new_c][w].add(u)
                
                idx = _edge_index(u, w, n)
                colors[idx] = new_c
        
        return colors, _verify_full(colors, n)
    
    def _precompute_path4(v, adj_list):
        """
        Precompute Path4[c][i][d] for all c, i, d < v.
        Path4[c][i][d] is True if there is a simple path of length 4 
        between i and d in the graph G_c induced by vertices 0..v-1.
        """
        Path4 = [ [[False]*v for _ in range(v)] for _ in range(4) ]
        for c in range(4):
            for i in range(v):
                if not adj_list[c][i]:
                    continue
                for d in range(v):
                    if i == d:
                        continue
                    found = False
                    # Path i-a-b-e-d
                    # a is neighbor of i
                    for a in adj_list[c][i]:
                        if a == d:
                            continue
                        # b is neighbor of a
                        for b in adj_list[c][a]:
                            if b == i or b == d or b == a:
                                continue
                            # e is neighbor of b
                            for e in adj_list[c][b]:
                                if e == i or e == d or e == a or e == b:
                                    continue
                                # Check if e is connected to d
                                if d in adj_list[c][e]:
                                    found = True
                                    break
                        if found:
                            break
                    Path4[c][i][d] = found
        return Path4

    def _constrained_backtrack(n_target, deadline):
        """
        Backtracking with strong constraint propagation using Path4.
        When adding vertex v, we check if assigning color c to edge (i, v)
        creates a C6 by looking for a path of length 4 between i and any
        existing neighbor d of v in color c.
        """
        adj = [[-1]*n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        
        def _search(v):
            if time.time() > deadline:
                return False
            
            if v == n_target:
                return True
            
            if v == 0:
                return _search(v + 1)
            
            # Precompute Path4 for the current subgraph 0..v-1
            # This is done once per level v
            Path4 = _precompute_path4(v, adj_list)
            
            def _assign(idx):
                if time.time() > deadline:
                    return False
                
                if idx == v:
                    return _search(v + 1)
                
                i = idx
                for c in range(4):
                    # Check if setting adj[i][v]=c creates a C6
                    # A C6 would involve v, i, and 4 other vertices in a path i-a-b-e-d
                    # where d is another neighbor of v in color c.
                    valid = True
                    for d in range(i):
                        if adj[d][v] == c:
                            if Path4[c][i][d]:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)
                    
                    if _assign(idx + 1):
                        return True
                    
                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)
                
                return False
            
            return _assign(0)
        
        if _search(0):
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj[i][j])
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
        
        # Try constrained backtracking first for smaller n
        if target_n <= 15:
            result = _constrained_backtrack(target_n, deadline - 20)
            if result is not None:
                if _verify_full(result, target_n):
                    best_n = target_n
                    best_colors = result
                    success = True
        
        if not success:
            # Try random + local search
            m = target_n * (target_n - 1) // 2
            max_restarts = 50 if target_n <= 10 else 20
            for restart in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                # Use a structured random initialization for better convergence
                if restart == 0:
                    # Try a block-based initialization
                    colors = []
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            # Use modular arithmetic for initial structure
                            c = (i * 3 + j * 5) % 4
                            colors.append(c)
                else:
                    colors = np.random.randint(0, 4, size=m).tolist()
                
                colors, valid = _local_search_fix(colors, target_n, deadline - 5, max_iters=5000)
                if valid:
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break
        
        if success:
            continue
        else:
            break
    
    # Final safety: ensure we have a valid result
    if best_n >= 6 and not _verify_full(best_colors, best_n):
        best_n = 5
        best_colors = [0] * (best_n*(best_n-1)//2)
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
