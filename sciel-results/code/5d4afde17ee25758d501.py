"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute: for each edge, store (u, v) as a tuple for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Build a mapping from vertex to its edges (as sorted tuples)
    vertex_edges = defaultdict(list)
    for u, v in edges:
        vertex_edges[u].append((min(u,v), max(u,v)))
        vertex_edges[v].append((min(u,v), max(u,v)))
    
    def check_bicolored_forest(colors, c1, c2):
        """Check if subgraph induced by colors c1 and c2 is a forest."""
        # Use union-find with path compression
        parent = list(range(n))
        rank = [0] * n
        
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
        
        for u, v in edges:
            if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                if not union(u, v):
                    return False
        return True
    
    def is_valid_full(colors):
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check bicolored forest constraint
        distinct = sorted(set(colors))
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if not check_bicolored_forest(colors, distinct[i], distinct[j]):
                    return False
        return True
    
    # Try 3-coloring with various patterns
    best_result = None
    best_num_colors = n + 1
    
    # Generate many candidate patterns
    patterns = []
    
    # Simple modular patterns
    for a in range(3):
        for b in range(3):
            patterns.append([(i * a + (i // 7) * b) % 3 for i in range(n)])
    patterns.append([i % 3 for i in range(n)])
    patterns.append([(i + (i // 5)) % 3 for i in range(n)])
    patterns.append([(i * 2 + (i // 10)) % 3 for i in range(n)])
    
    # Try to exploit the graph structure
    # The graph appears to have a regular structure - let's try row/column based
    for offset in range(3):
        patterns.append([(i // 5 + offset) % 3 for i in range(n)])
        patterns.append([(i % 5 + i // 5 + offset) % 3 for i in range(n)])
    
    for pat in patterns:
        if time.time() > deadline:
            break
        # Quick proper coloring check first
        proper = True
        for u, v in edges:
            if pat[u] == pat[v]:
                proper = False
                break
        if proper and is_valid_full(pat):
            best_result = pat
            best_num_colors = 3
            break
    
    if best_result is None:
        # Backtracking with 3 colors using incremental constraint checking
        # Order vertices by degree (descending)
        vertex_order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos_in_order = {v: i for i, v in enumerate(vertex_order)}
        
        # Precompute for each vertex, which neighbors come before it in the ordering
        # This helps with constraint propagation
        
        colors = [None] * n
        
        # Track for each color pair (c1, c2), the set of vertices with those colors
        # and use incremental cycle detection
        # For efficiency, we'll check only the pairs that are affected
        
        def bt3(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return is_valid_full(colors)
            
            v = vertex_order[idx]
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    forbidden.add(colors[nb])
            
            for c in range(3):
                if c in forbidden:
                    continue
                
                colors[v] = c
                # Check if this creates a bicolored cycle
                # Only need to check pairs (c, c2) for c2 in {0,1,2}
                # But we can be smarter: check if adding v with color c creates a cycle
                # in the subgraph induced by colors c and c2
                valid = True
                for c2 in range(3):
                    if c2 == c:
                        continue
                    # Check if the edge (v, nb) where nb has color c2 creates a cycle
                    # in the subgraph of colors {c, c2}
                    for nb in adj[v]:
                        if colors[nb] == c2:
                            # Check if there's already a path from v to nb using only
                            # vertices with colors c or c2 (excluding the direct edge)
                            # This is expensive, so let's just do the full check for this pair
                            if not check_bicolored_forest(colors, c, c2):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    if bt3(idx + 1):
                        return True
                
                colors[v] = None
            
            return False
        
        if bt3(0):
            best_result = colors
            best_num_colors = 3
    
    if best_result is None:
        # Try 4 colors with backtracking
        vertex_order = sorted(range(n), key=lambda x: -len(adj[x]))
        colors = [None] * n
        
        def bt4(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return is_valid_full(colors)
            
            v = vertex_order[idx]
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    forbidden.add(colors[nb])
            
            for c in range(4):
                if c in forbidden:
                    continue
                
                colors[v] = c
                valid = True
                for c2 in range(4):
                    if c2 == c:
                        continue
                    if not check_bicolored_forest(colors, c, c2):
                        valid = False
                        break
                
                if valid:
                    if bt4(idx + 1):
                        return True
                
                colors[v] = None
            
            return False
        
        if bt4(0):
            best_result = colors
            best_num_colors = 4
    
    if best_result is None:
        # Fallback: greedy coloring with many colors
        colors = [None] * n
        for v in range(n):
            used = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    used.add(colors[nb])
            for c in range(n):
                if c not in used:
                    colors[v] = c
                    break
        best_result = colors
        best_num_colors = max(colors) + 1
    
    # Ensure all are Python ints
    best_result = [int(c) for c in best_result]
    return best_result

# EVOLVE-BLOCK-END
