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
        
    # Order vertices by degree descending for better backtracking pruning
    vertex_order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Map vertex index to its position in the order
    # Actually, we just iterate through vertex_order
    
    def backtrack(k, idx, coloring):
        if idx == n:
            return True
        
        v = vertex_order[idx]
        
        # Try each color from 0 to k-1
        for c in range(k):
            # 1. Check proper coloring: no neighbor has same color
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # 2. Check bicolored forest constraint
            # For each other color d, check if adding v (color c) creates a cycle in {c, d} subgraph
            valid = True
            for d in range(k):
                if d == c:
                    continue
                
                # Get neighbors of v with color c or d
                neigh_c = []
                neigh_d = []
                for u in adj[v]:
                    if coloring[u] == c:
                        neigh_c.append(u)
                    elif coloring[u] == d:
                        neigh_d.append(u)
                    
                if not neigh_c and not neigh_d:
                    continue
                
                # We need to check if any two neighbors in (neigh_c + neigh_d) are connected
                # in the subgraph induced by vertices with color c or d.
                # Since the subgraph is a forest, we can find the root of each neighbor.
                # If two neighbors have the same root, adding v creates a cycle.
                
                # To find roots, we need the current forest structure for {c, d}.
                # We can compute it on the fly.
                # Vertices in {c, d} subgraph:
                cd_verts = set()
                for i in range(n):
                    if coloring[i] == c or coloring[i] == d:
                        cd_verts.add(i)
                
                if not cd_verts:
                    continue
                
                # Union-Find for this specific pair
                parent = {x: x for x in cd_verts}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx == ry:
                        return False
                    parent[ry] = rx
                    return True
                
                # Build the forest
                for u in cd_verts:
                    for w in adj[u]:
                        if w in cd_verts and u < w:
                            union(u, w)
                
                # Check neighbors
                roots = set()
                for u in neigh_c + neigh_d:
                    roots.add(find(u))
                
                if len(roots) < len(neigh_c + neigh_d):
                    valid = False
                    break
            
            if not valid:
                continue
            
            coloring[v] = c
            if backtrack(k, idx + 1, coloring):
                return True
            coloring[v] = -1
        
        return False

    # Try increasing number of colors
    for k in range(2, 6): # Likely 4 or 5 colors
        coloring = [-1] * n
        if backtrack(k, 0, coloring):
            return [int(c) for c in coloring]
    
    # Fallback
    return list(range(n))

# EVOLVE-BLOCK-END
