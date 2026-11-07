"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def solve():
        # Try with increasing number of colors
        for num_colors in range(3, 11):
            result = _solve_with_k_colors(num_colors)
            if result is not None:
                return result
        return None
    
    def _solve_with_k_colors(k):
        # Backtracking with DSU per color pair and undo/redo
        deg = [len(adj[i]) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda x: -deg[x])
        
        assign = [-1] * n
        
        # DSU per color pair: dsu[c1][c2] where c1 < c2
        # Each DSU has: parent dict, rank dict
        # We track operations for undo
        
        # For efficiency, use a flat index for pairs
        num_pairs = k * (k - 1) // 2
        pair_index = {}
        idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                pair_index[(i, j)] = idx
                idx += 1
        
        # DSU state: list of (parent, rank) dicts for each pair
        dsu_parents = [dict() for _ in range(num_pairs)]
        dsu_ranks = [dict() for _ in range(num_pairs)]
        
        # Operation log for undo: list of (pair_idx, x, y, rx, ry, rank_changed)
        op_log = []
        
        def dsu_find(pi, x):
            parent = dsu_parents[pi]
            if x not in parent:
                parent[x] = x
                dsu_ranks[pi][x] = 0
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def dsu_same(pi, x, y):
            return dsu_find(pi, x) == dsu_find(pi, y)
        
        def dsu_union(pi, x, y):
            """Union x and y in DSU pi. Returns None if already connected, else operation info for undo."""
            parent = dsu_parents[pi]
            rank = dsu_ranks[pi]
            
            # Ensure x and y are in the DSU
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            if y not in parent:
                parent[y] = y
                rank[y] = 0
            
            rx = dsu_find(pi, x)
            ry = dsu_find(pi, y)
            
            if rx == ry:
                return None  # Already connected, would create cycle
            
            # Union by rank
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            rank_changed = False
            if rank[rx] == rank[ry]:
                rank[rx] += 1
                rank_changed = True
            
            # Record operation for undo
            op_log.append((pi, ry, rx, rank_changed))
            return True
        
        def undo_last_ops(count):
            """Undo the last `count` union operations."""
            for _ in range(count):
                if not op_log:
                    break
                pi, ry, rx, rank_changed = op_log.pop()
                parent = dsu_parents[pi]
                rank = dsu_ranks[pi]
                # Restore: ry was child of rx, now make ry its own root
                parent[ry] = ry
                if rank_changed:
                    rank[rx] -= 1
        
        def backtrack(idx):
            if idx == len(order):
                return True
            
            v = order[idx]
            
            for c in range(k):
                # Check if color c conflicts with any neighbor
                valid = True
                for u in adj[v]:
                    if assign[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign v to color c
                assign[v] = c
                
                # For each neighbor u with a different color c2, check DSU for pair (c, c2)
                ok = True
                ops_made = 0
                for u in adj[v]:
                    c2 = assign[u]
                    if c2 == -1 or c2 == c:
                        continue
                    
                    # Get pair index
                    ci, cj = (c, c2) if c < c2 else (c2, c)
                    pi = pair_index[(ci, cj)]
                    
                    # Check if u and v are already connected in this pair's DSU
                    if dsu_same(pi, u, v):
                        ok = False
                        break
                    
                    # Union them
                    dsu_union(pi, u, v)
                    ops_made += 1
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                assign[v] = -1
                undo_last_ops(ops_made)
            
            return False
        
        if backtrack(0):
            return assign
        return None
    
    result = solve()
    if result is None:
        # Fallback: return unique colors (always valid but not optimal)
        result = list(range(n))
    
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
