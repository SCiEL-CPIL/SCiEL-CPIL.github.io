import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(coloring, n, edges):
    """
    Checks if the partial coloring is valid for the assigned vertices.
    Returns True if valid, False otherwise.
    """
    # 1. Proper coloring check
    for u, v in edges:
        if coloring[u] != -1 and coloring[v] != -1:
            if coloring[u] == coloring[v]:
                return False

    # 2. Bicolored forest check
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != -1 and cv != -1 and cu != cv:
            key = (min(cu, cv), max(cu, cv))
            pair_edges[key].append((u, v))
            
    for (c1, c2), edge_list in pair_edges.items():
        # Union-Find to check for cycles
        parent = {}
        rank = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
            
        has_cycle = False
        for u, v in edge_list:
            if not union(u, v):
                has_cycle = True
                break
        
        if has_cycle:
            return False
            
    return True

def _backtrack_search(n, edges, k, order, adj, deadline):
    """
    Backtracking search for a valid k-coloring.
    Returns a list of colors if found, None otherwise.
    """
    coloring = [-1] * n
    steps = 0
    max_steps = 50000 # Safety limit to prevent infinite loops in bad cases
    
    def backtrack(idx):
        nonlocal steps
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        v = order[idx]
        
        # Get available colors
        used_colors = set()
        for u in adj[v]:
            if coloring[u] != -1:
                used_colors.add(coloring[u])
                
        # Try each color
        for c in range(k):
            if c in used_colors:
                continue
                
            coloring[v] = c
            steps += 1
            
            # Check validity of the partial solution
            # Optimization: Only check constraints involving v?
            # Full check is O(E) which is ~100. 50*100*50*50 = 12.5M ops. Might be tight in Python.
            # Let's try a localized check first.
            
            # Check proper coloring for neighbors
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            
            if valid:
                # Check bicolored cycles involving v
                # We only need to check pairs (c, cu) where cu is color of a neighbor
                for u in adj[v]:
                    cu = coloring[u]
                    if cu == -1 or cu == c:
                        continue
                    # Check if adding (v,u) to pair (min(c,cu), max(c,cu)) creates a cycle
                    # We need to check connectivity in the subgraph induced by this pair
                    # This is expensive to do incrementally without state.
                    # Given the small N, let's just do the full check on the partial assignment.
                    # It's safer and simpler.
                    pass
                    
            if valid:
                if _check_validity(coloring, n, edges):
                    if backtrack(idx + 1):
                        return True
            
            coloring[v] = -1
            
        return False

    if backtrack(0):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and degrees
    adj = defaultdict(list)
    degrees = [0] * n
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1
        
    # Order vertices by degree (descending) to fail fast
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Try increasing number of colors
    for k in range(4, n + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts with different orderings if needed?
        # For now, try the fixed degree-ordered backtracking.
        # If it's too slow, we might need to randomize the order slightly.
        
        result = _backtrack_search(n, edges, k, order, adj, deadline)
        
        if result is not None:
            # Verify the result one last time
            if _check_validity(result, n, edges):
                return result
                
    # Fallback: A valid coloring with N colors (identity)
    # This is always valid: no adjacent same color, and any pair of colors has at most 1 edge (forest)
    return list(range(n))
