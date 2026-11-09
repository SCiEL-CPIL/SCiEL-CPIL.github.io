"""Rewritten program: ILP-based search for 4-colorings of K_n with no monochromatic C6."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s kill
    
    # Precompute all distinct 6-cycles on 6 labeled vertices
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
    
    def _find_c6(adj, n):
        """Find a C6 in any color class. Returns (color, subset, cycle) or None."""
        cycles = _get_cycles()
        for c in range(4):
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
        adj, _ = _build_adj_structures(colors, n)
        return _find_c6(adj, n) is None
    
    def _ilp_solve(n, deadline, max_time=30):
        """
        Use a simple backtracking ILP-like approach with constraint propagation.
        Variables: color of each edge in K_n, domain {0,1,2,3}.
        Constraints: no monochromatic C6.
        """
        m = n * (n - 1) // 2
        if m == 0:
            return []
        
        # Edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_to_idx = {e: idx for idx, e in enumerate(edges)}
        
        # Precompute which edges form C6s
        # For each C6 pattern on 6 vertices, the 6 edges of the cycle
        cycles = _get_cycles()
        c6_edge_sets = []
        seen_c6 = set()
        for subset in combinations(range(n), 6):
            for cycle in cycles:
                edge_set = frozenset(
                    frozenset((subset[cycle[k]], subset[cycle[(k+1)%6]])) 
                    for k in range(6)
                )
                if edge_set not in seen_c6:
                    seen_c6.add(edge_set)
                    c6_edge_sets.append(edge_set)
        
        # For each edge, list of C6s it participates in
        edge_to_c6s = [[] for _ in range(m)]
        for c6_idx, edge_set in enumerate(c6_edge_sets):
            for e in edge_set:
                edge_to_c6s[edge_to_idx[tuple(sorted(e))]].append(c6_idx)
        
        # Backtracking with forward checking
        colors = [-1] * m
        # For each C6, track how many of its 6 edges are assigned to each color
        c6_color_counts = [[0, 0, 0, 0] for _ in c6_edge_sets]
        
        def _is_c6_forbidden(c6_idx, edge_idx, color):
            """Check if assigning edge_idx to color would make c6_idx all same color."""
            if c6_color_counts[c6_idx][color] + 1 == 6:
                return True
            return False
        
        def _search(edge_pos):
            if time.time() > deadline:
                return False
            
            if edge_pos == m:
                return True
            
            e = edges[edge_pos]
            ei = edge_to_idx[e]
            
            # Try each color
            for c in range(4):
                # Check if this assignment would create a C6
                valid = True
                for c6_idx in edge_to_c6s[ei]:
                    if _is_c6_forbidden(c6_idx, ei, c):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                colors[ei] = c
                for c6_idx in edge_to_c6s[ei]:
                    c6_color_counts[c6_idx][c] += 1
                
                if _search(edge_pos + 1):
                    return True
                
                # Unassign
                for c6_idx in edge_to_c6s[ei]:
                    c6_color_counts[c6_idx][c] -= 1
                colors[ei] = -1
            
            return False
        
        if _search(0):
            return colors
        return None
    
    def _local_search(colors, n, deadline, max_iters=5000):
        """Local search to fix violations by flipping edges."""
        adj, adj_list = _build_adj_structures(colors, n)
        cycles = _get_cycles()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find a violating C6
            result = _find_c6(adj, n)
            if result is None:
                return colors, True
            
            c, subset, cycle = result
            
            # Try to fix by flipping one edge in the cycle
            best_flip = None
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                if u > w:
                    u, w = w, u
                
                old_c = adj[u][w]
                
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    
                    # Temporarily flip
                    adj_list[old_c][u].discard(w)
                    adj_list[old_c][w].discard(u)
                    adj[u][w] = new_c
                    adj[w][u] = new_c
                    adj_list[new_c][u].add(w)
                    adj_list[new_c][w].add(u)
                    
                    # Check if current C6 is fixed
                    still_has_current = False
                    for ck in range(6):
                        uu = subset[cycle[ck]]
                        ww = subset[cycle[(ck+1)%6]]
                        if adj[uu][ww] != c:
                            still_has_current = True
                            break
                    
                    if still_has_current:
                        # Undo
                        adj_list[new_c][u].discard(w)
                        adj_list[new_c][w].discard(u)
                        adj[u][w] = old_c
                        adj[w][u] = old_c
                        adj_list[old_c][u].add(w)
                        adj_list[old_c][w].add(u)
                        continue
                    
                    # Accept this flip
                    idx = _edge_index(u, w, n)
                    colors[idx] = new_c
                    best_flip = (u, w, old_c, new_c)
                    break
                
                if best_flip:
                    break
            
            if best_flip is None:
                # Random flip
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
    
    # Start with small n and try to grow
    best_n = 5
    best_colors = [0] * (best_n*(best_n-1)//2)
    
    # Try to construct valid colorings for increasing n
    for target_n in range(6, 21):
        if time.time() > deadline - 10:
            break
        
        success = False
        
        # Try ILP/backtracking for smaller n
        if target_n <= 12:
            result = _ilp_solve(target_n, deadline - 20)
            if result is not None:
                if _verify_full(result, target_n):
                    best_n = target_n
                    best_colors = result
                    success = True
        
        if not success:
            # Try random + local search
            m = target_n * (target_n - 1) // 2
            max_restarts = 30 if target_n <= 10 else 15
            for restart in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                # Structured random initialization
                if restart == 0:
                    colors = []
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            c = (i * 3 + j * 5) % 4
                            colors.append(c)
                else:
                    colors = np.random.randint(0, 4, size=m).tolist()
                
                colors, valid = _local_search(colors, target_n, deadline - 5, max_iters=3000)
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
