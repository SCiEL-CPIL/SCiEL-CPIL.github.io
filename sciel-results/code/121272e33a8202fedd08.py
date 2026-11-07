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
        
    def is_valid_coloring(coloring):
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored forest constraint using union-find
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                verts = [i for i in range(n) if coloring[i] in (c1, c2)]
                if len(verts) <= 1:
                    continue
                vert_set = set(verts)
                parent = {v: v for v in verts}
                rank = {v: 0 for v in verts}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    if rank[rx] < rank[ry]:
                        rx, ry = ry, rx
                    parent[ry] = rx
                    if rank[rx] == rank[ry]:
                        rank[rx] += 1
                    return True
                
                for u in verts:
                    for v in adj[u]:
                        if v in vert_set and u < v:
                            if not union(u, v):
                                return False
        return True

    # Try 3 colors with the (r+c)%3 pattern for 0-48, and assign 49 carefully
    # Grid is 7x7 for 0-48.
    # v = 7*r + c. r in 0..6, c in 0..6.
    coloring_3 = [-1] * n
    for v in range(49):
        r = v // 7
        c = v % 7
        coloring_3[v] = (r + c) % 3
        
    # Vertex 49 is connected to 0, 6, 48.
    # Colors: 0:0, 6:0, 48:0.
    # Try 1 for 49.
    coloring_3[49] = 1
    if is_valid_coloring(coloring_3):
        return [int(c) for c in coloring_3]
    
    # Try 2 for 49.
    coloring_3[49] = 2
    if is_valid_coloring(coloring_3):
        return [int(c) for c in coloring_3]
        
    # If 3 colors failed, try 4 colors.
    # Use (r+c)%4 for grid?
    coloring_4 = [-1] * n
    for v in range(49):
        r = v // 7
        c = v % 7
        coloring_4[v] = (r + c) % 4
        
    # Assign 49. Neighbors 0(0), 6(6%4=2), 48(12%4=0).
    # 49 cannot be 0 or 2. Try 1.
    coloring_4[49] = 1
    if is_valid_coloring(coloring_4):
        return [int(c) for c in coloring_4]
        
    coloring_4[49] = 3
    if is_valid_coloring(coloring_4):
        return [int(c) for c in coloring_4]
        
    # Fallback to backtracking with 4 colors if patterns fail
    vertex_order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def backtrack(k, idx, coloring):
        if idx == n:
            return True
        v = vertex_order[idx]
        for c in range(k):
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Simple check for bicolored cycle with assigned neighbors
            # This is an approximation, but combined with final check it should be ok
            # To be safe, we just assign and check at the end in a loop?
            # No, backtracking needs pruning.
            # We'll just use the final check in the loop of k, but that's slow.
            # For now, rely on the patterns.
            
            coloring[v] = c
            if backtrack(k, idx + 1, coloring):
                return True
            coloring[v] = -1
        return False

    for k in range(4, n + 1):
        coloring = [-1] * n
        if backtrack(k, 0, coloring):
            if is_valid_coloring(coloring):
                return [int(c) for c in coloring]
                
    return list(range(n))

# EVOLVE-BLOCK-END
