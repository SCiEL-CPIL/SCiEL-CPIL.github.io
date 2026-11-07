"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(colors, edges):
    """Check if coloring is valid: adjacent differ and no bicolored cycles."""
    n = len(colors)
    # Check adjacent vertices have different colors
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    # Check no bicolored cycles using union-find per color pair
    # For each pair of colors, build subgraph and check if it's a forest
    from collections import defaultdict
    
    # Union-Find class
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False  # cycle detected
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    # Group edges by color pair
    edge_color_pairs = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu > cv:
            cu, cv = cv, cu
        edge_color_pairs[(cu, cv)].append((u, v))
    
    # For each color pair, check if the induced subgraph is a forest
    for (c1, c2), edge_list in edge_color_pairs.items():
        # Only vertices with color c1 or c2 matter
        # Use union-find to detect cycles
        uf_parent = list(range(n))
        uf_rank = [0] * n
        
        def uf_find(x):
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x
        
        def uf_union(x, y):
            px, py = uf_find(x), uf_find(y)
            if px == py:
                return False
            if uf_rank[px] < uf_rank[py]:
                px, py = py, px
            uf_parent[py] = px
            if uf_rank[px] == uf_rank[py]:
                uf_rank[px] += 1
            return True
        
        for u, v in edge_list:
            if not uf_union(u, v):
                return False  # cycle found
    
    return True

def _greedy_coloring(edges, n, max_colors):
    """Try to color the graph with at most max_colors using greedy + backtracking."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    colors = [-1] * n
    colors[0] = 0
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Try colors in order
        for c in range(max_colors):
            # Check if color c is valid for vertex idx
            valid = True
            for neighbor in adj[idx]:
                if colors[neighbor] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            colors[idx] = c
            
            # Check if adding this edge creates any bicolored cycle
            # We need to check all edges involving vertex idx
            # Actually, we should check after assigning, but that's expensive
            # Let's just assign and check at the end, or check incrementally
            
            if backtrack(idx + 1):
                return True
        
        colors[idx] = -1
        return False
    
    if backtrack(0):
        return colors
    return None

def _search_optimal(edges, n):
    """Search for the minimum number of colors."""
    # Try 2 colors first
    for num_colors in range(2, 10):
        colors = _greedy_coloring(edges, n, num_colors)
        if colors is not None:
            if _check_validity(colors, edges):
                return colors, num_colors
    return None, None

def construct():
    n = P['n']
    edges = P['edges']
    
    # First, try to find an optimal or near-optimal coloring
    best_colors = None
    best_num = None
    
    # Try small numbers of colors
    for num_colors in range(2, 8):
        colors = _greedy_coloring(edges, n, num_colors)
        if colors is not None:
            if _check_validity(colors, edges):
                best_colors = colors
                best_num = num_colors
                break
    
    if best_colors is None:
        # Fallback: use all distinct colors (always valid)
        best_colors = list(range(n))
        best_num = n
    
    # Ensure we return Python ints
    result = [int(c) for c in best_colors]
    return result
# EVOLVE-BLOCK-END
