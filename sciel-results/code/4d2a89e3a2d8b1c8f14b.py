import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']

# Precompute adjacency
adj = [[] for _ in range(N)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute degree
degree = [len(adj[i]) for i in range(N)]

def _is_valid(colors):
    """Check if a coloring is valid: proper and no bicolored cycles."""
    # Check properness
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Check no bicolored cycle using union-find per color pair
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        a, b = colors[u], colors[v]
        if a != b:
            key = (min(a, b), max(a, b))
            pair_edges[key].append((u, v))
    
    for key, elist in pair_edges.items():
        # Union-find on vertices in this bicolored subgraph
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Try to find a valid coloring with as few colors as possible
    # Start with 2 colors (bipartite check)
    # The graph might not be bipartite, so try 3, 4, 5, ...
    
    best_colors = None
    best_count = N + 1
    
    # First, check if graph is bipartite (2-colorable)
    # Then check 3, 4, 5, ... colors
    
    # Use backtracking with forward checking for small number of colors
    # For each k from 2 upward, try to find a valid k-coloring
    
    # Variable ordering: by degree (highest first) for better pruning
    var_order = sorted(range(N), key=lambda v: -degree[v])
    
    for k in range(2, 10):  # Try up to 9 colors
        if time.time() > deadline:
            break
        if k >= best_count:
            break
        
        # Backtracking search
        colors = [-1] * N
        # Union-find structures for each color pair
        # We need to track connectivity for each (a,b) pair
        # Use a dict mapping (a,b) -> dict of parent for union-find
        
        # For efficiency, we'll track for each pair the union-find state
        # But this is complex. Let's use a simpler approach:
        # Assign colors one by one, and after each assignment, check if it creates a bicolored cycle
        
        # Actually, let's use a different approach:
        # For each color pair (a,b), maintain the connected components of the subgraph induced by edges with colors a and b.
        # When we assign a color to a vertex, we add edges to the relevant pairs and check for cycles.
        
        # This is complex. Let's use a simpler backtracking with validation.
        
        # For small k, we can use backtracking with pruning.
        # Order variables by degree.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == N:
                # All vertices colored, verify
                if _is_valid(colors):
                    return True
                return False
            
            v = var_order[idx]
            
            # Try each color
            for c in range(k):
                # Check if c is valid for v (not used by neighbors)
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            
            return False
        
        if backtrack(0):
            distinct = _count_distinct(colors)
            if distinct < best_count:
                best_count = distinct
                best_colors = colors[:]
            # If we found with k colors, and k == distinct, we can stop if k is minimal
            if best_count <= k:
                break
    
    if best_colors is not None:
        # Ensure we return Python ints
        return [int(c) for c in best_colors]
    
    # Fallback: greedy coloring
    colors = [0] * N
    for v in var_order:
        used = set(colors[u] for u in adj[v])
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    
    if _is_valid(colors):
        return [int(c) for c in colors]
    
    # Absolute fallback: identity
    return [int(i) for i in range(N)]
