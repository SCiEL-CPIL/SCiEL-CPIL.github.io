"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest_fast(colors, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest.
    Returns True if valid, False otherwise."""
    # Group vertices by color
    color_groups = defaultdict(set)
    for i in range(n):
        color_groups[colors[i]].add(i)
    
    colors_used = list(color_groups.keys())
    
    for idx_c1 in range(len(colors_used)):
        for idx_c2 in range(idx_c1 + 1, len(colors_used)):
            c1, c2 = colors_used[idx_c1], colors_used[idx_c2]
            vertices = color_groups[c1] | color_groups[c2]
            if len(vertices) <= 1:
                continue
            
            # Union-Find
            parent = {}
            for v in vertices:
                parent[v] = v
            
            def find(x):
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
            
            for u in vertices:
                for v in adj[u]:
                    if v in vertices and u < v:
                        if not union(u, v):
                            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _is_valid_coloring(colors, adj, n):
    """Check proper coloring and bicolored forest constraint."""
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    return _check_bicolor_forest_fast(colors, adj, n)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    adj = _build_adjacency()
    
    # Precompute edge list for fast access
    edges = P['edges']
    
    # Strategy: Try small numbers of colors (3, 4, 5, ...) with backtracking
    # For each k, try to assign colors 0..k-1 to vertices such that:
    # 1. Adjacent vertices have different colors
    # 2. For any pair of colors, the induced subgraph is a forest
    
    # First, try a structured approach based on the graph's apparent grid structure
    # The graph looks like a 5x10 grid with extra edges
    
    best_result = None
    best_score = n + 1
    
    # Try k=3 first
    for k in range(3, 12):
        if time.time() > deadline:
            break
        
        # Try multiple random initial assignments and local search
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initial proper coloring using k colors
            import random
            random.seed(attempt * 1000 + k)
            colors = [0] * n
            
            # Greedy random assignment
            for i in range(n):
                used = set()
                for v in adj[i]:
                    if v < i:
                        used.add(colors[v])
                avail = [c for c in range(k) if c not in used]
                if avail:
                    colors[i] = random.choice(avail)
                else:
                    # Shouldn't happen for proper k-coloring if k >= chromatic number
                    colors[i] = 0
                    break
            
            # Check if proper
            proper = True
            for u, v in edges:
                if colors[u] == colors[v]:
                    proper = False
                    break
            if not proper:
                continue
            
            # Check bicolored forest
            if not _check_bicolor_forest_fast(colors, adj, n):
                continue
            
            # Valid! Try to reduce colors via local search
            current = colors[:]
            distinct = _count_distinct(current)
            
            # Local search to minimize distinct colors
            for _iter in range(200):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    old_c = current[i]
                    # Try each color 0..k-1
                    for new_c in range(k):
                        if new_c == old_c:
                            continue
                        # Check proper coloring constraint
                        proper = True
                        for v in adj[i]:
                            if current[v] == new_c:
                                proper = False
                                break
                        if not proper:
                            continue
                        
                        # Tentative assignment
                        current[i] = new_c
                        if _check_bicolor_forest_fast(current, adj, n):
                            new_distinct = _count_distinct(current)
                            if new_distinct < distinct:
                                distinct = new_distinct
                                improved = True
                                break
                            else:
                                current[i] = old_c
                        else:
                            current[i] = old_c
                    if improved:
                        break
                if not improved:
                    break
            
            score = _count_distinct(current)
            if score < best_score:
                best_score = score
                best_result = current[:]
                found = True
            
            if best_score <= k:
                break
        
        if found and best_score <= k:
            break
    
    # If nothing found, fall back to trivial solution
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
