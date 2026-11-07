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
    
    # For a given coloring, check if it's valid:
    # 1. Proper: adjacent vertices have different colors
    # 2. For every pair of colors (c1, c2), the subgraph induced by vertices with color c1 or c2 is a forest
    def is_valid_coloring(coloring):
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored forest constraint using union-find
        # For each pair of colors, check if the induced subgraph is a forest
        colors_used = set(coloring)
        color_pairs = set()
        for c in colors_used:
            for d in colors_used:
                if c < d:
                    color_pairs.add((c, d))
        
        for c1, c2 in color_pairs:
            # Get vertices with color c1 or c2
            verts = [i for i in range(n) if coloring[i] in (c1, c2)]
            if len(verts) <= 1:
                continue
            
            # Build a local index map
            vert_set = set(verts)
            # Union-find on these vertices
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
                    return False  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Add edges between vertices in verts
            for u in verts:
                for v in adj[u]:
                    if v in vert_set and u < v:
                        if not union(u, v):
                            return False  # cycle found in bicolored subgraph
        
        return True
    
    # Try to find coloring with k colors, starting from small k
    # Use backtracking with ordering by degree (highest degree first)
    # Order vertices by degree descending
    vertex_order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    # We'll assign colors in the order of vertex_order
    # coloring[i] = color of vertex i, -1 means unassigned
    def backtrack(k, idx, coloring):
        if idx == n:
            return True
        
        v = vertex_order[idx]
        
        # Try each color from 0 to k-1
        for c in range(k):
            # Check proper coloring: no neighbor has same color
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored forest constraint with existing assignments
            # For each color d already used, check if adding v with color c
            # creates a bicolored cycle with color d
            valid = True
            for d in range(k):
                if d == c:
                    continue
                # Check if there's a path between any two neighbors of v 
                # that have color d, using only vertices with color c or d
                # Actually, we need to check: if we add v (color c) to the subgraph 
                # of colors {c, d}, does it create a cycle?
                # This happens if v has two neighbors with color d that are 
                # already connected in the {c,d} subgraph, OR v has two neighbors 
                # with color c that are connected, OR one neighbor with color c 
                # and one with color d that are connected.
                
                # Collect neighbors of v with color c or d
                neigh_c = [u for u in adj[v] if coloring[u] == c]
                neigh_d = [u for u in adj[v] if coloring[u] == d]
                
                if not neigh_c and not neigh_d:
                    continue
                
                # We need to check if adding v creates a cycle in the {c,d} subgraph
                # Build union-find on the current {c,d} subgraph
                cd_verts = set()
                for i in range(n):
                    if coloring[i] in (c, d):
                        cd_verts.add(i)
                
                if len(cd_verts) == 0:
                    continue
                
                parent = {x: x for x in cd_verts}
                rank = {x: 0 for x in cd_verts}
                
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
                
                # Add all existing edges in {c,d} subgraph
                for u in cd_verts:
                    for w in adj[u]:
                        if w in cd_verts and u < w:
                            union(u, w)
                
                # Now try to add v
                # v connects to all its neighbors in cd_verts
                # A cycle is created if v has two neighbors in the same connected component
                components = set()
                for u in neigh_c + neigh_d:
                    components.add(find(u))
                
                if len(components) < len(neigh_c + neigh_d):
                    # Two neighbors in same component -> cycle
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
    for k in range(2, n + 1):
        coloring = [-1] * n
        if backtrack(k, 0, coloring):
            # Convert to proper Python ints
            result = [int(c) for c in coloring]
            return result
    
    # Fallback: use n colors (trivially valid)
    return list(range(n))

# EVOLVE-BLOCK-END
