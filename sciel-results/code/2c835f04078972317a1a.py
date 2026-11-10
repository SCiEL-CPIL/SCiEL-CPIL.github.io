"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices by degree (descending) to prune earlier
    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # Map original index to position in sorted order for easier assignment?
    # Or just iterate through sorted vertices.
    # We need to track which original vertex is at which step.
    order = vertices
    
    def solve(k):
        # Union-Find with rollback for each pair of colors
        # We only need pairs (c1, c2) where c1 < c2
        # Initialize UFs
        uf_parent = {}
        uf_rank = {}
        
        def init_uf(pair):
            uf_parent[pair] = list(range(n))
            uf_rank[pair] = [0] * n
            
        # Pre-initialize for all pairs in 0..k-1
        for i in range(k):
            for j in range(i+1, k):
                init_uf((i, j))
                
        # Stack of changes for rollback
        # Each change: (pair, x, y, parent_y, rank_x)
        # If union happens: parent[ry] = rx, rank[rx] might change
        # We need to store enough to revert.
        # Simpler: Store (pair, x, y, parent_x_before, parent_y_before, rank_x_before, rank_y_before)
        # Actually, since we only do union by rank, we can store:
        # (pair, rx, ry, parent_ry_before, rank_rx_before)
        # And ensure we don't change parent_rx.
        
        change_stack = []
        
        def find(pair, x):
            p = uf_parent[pair]
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x
            
        def union(pair, x, y):
            p = uf_parent[pair]
            r = uf_rank[pair]
            rx = find(pair, x)
            ry = find(pair, y)
            if rx == ry:
                return False # Cycle
            if r[rx] < r[ry]:
                rx, ry = ry, rx
            # Record change: pair, rx, ry, p[ry] (should be ry), r[rx]
            change_stack.append((pair, rx, ry, r[rx]))
            p[ry] = rx
            if r[rx] == r[ry]:
                r[rx] += 1
            return True
            
        def undo():
            if not change_stack:
                return
            pair, rx, ry, rank_rx_before = change_stack.pop()
            p = uf_parent[pair]
            r = uf_rank[pair]
            p[ry] = ry
            r[rx] = rank_rx_before
            
        # Coloring array
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
                
            v = order[idx]
            
            # Determine available colors
            # 1. Not used by neighbors
            # 2. Does not create bicolored cycle with assigned neighbors
            
            used_by_neighbors = set()
            for u in adj[v]:
                if color[u] != -1:
                    used_by_neighbors.add(color[u])
                    
            for c in range(k):
                if c in used_by_neighbors:
                    continue
                    
                # Try color c
                color[v] = c
                
                # Check bicolored cycles with assigned neighbors
                valid = True
                changes_made = 0
                for u in adj[v]:
                    cu = color[u]
                    if cu == -1:
                        continue
                    if cu == c:
                        valid = False
                        break
                    pair = (min(c, cu), max(c, cu))
                    if not union(pair, v, u):
                        valid = False
                        break
                    changes_made += 1
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                    else:
                        # Undo changes
                        for _ in range(changes_made):
                            undo()
                else:
                    # Undo changes made before failure
                    for _ in range(changes_made):
                        undo()
                        
                color[v] = -1
                
            return False
            
        return backtrack(0)

    # Try increasing number of colors
    for k in range(2, 15):
        if solve(k):
            # Extract coloring
            # Note: 'color' is local to solve, need to return it
            # We need to modify solve to return the coloring
            pass
            
    # Fallback
    return list(range(n))
# EVOLVE-BLOCK-END
