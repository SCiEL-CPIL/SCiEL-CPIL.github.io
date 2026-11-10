"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _validate_full(coloring, edges, n):
    """Full validation: proper coloring + no bicolored cycles."""
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check bicolored cycles
    colors_used = sorted(set(coloring))
    for i, c1 in enumerate(colors_used):
        for c2 in colors_used[i+1:]:
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
                    return True  # cycle
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return False
            
            for u, v in edges:
                if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                    if union(u, v):
                        return False
    return True


def _four_color_grid(n):
    """4-coloring using (row%2)*2 + (col%2) on 5x10 grid."""
    rows, cols = 5, 10
    coloring = []
    for i in range(n):
        row = i // cols
        col = i % cols
        coloring.append(int((row % 2) * 2 + (col % 2)))
    return coloring


def _check_bicolored_incremental(colors, edges, n, new_vertex, new_color):
    """Check if assigning new_color to new_vertex creates a bicolored cycle.
    Only checks pairs involving new_color for efficiency."""
    for c2 in range(new_color + 1):
        if c2 == new_color:
            continue
        # Check subgraph induced by {new_color, c2}
        # Collect vertices with these colors
        in_sub = set()
        for i in range(n):
            if colors[i] == new_color or colors[i] == c2:
                in_sub.add(i)
        
        if len(in_sub) < 3:
            continue
        
        parent = {v: v for v in in_sub}
        
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
        
        for u, v in edges:
            if u in parent and v in parent:
                if not union(u, v):
                    return False
    return True


def _backtrack_3color(edges, adj, n, vertex_order, deadline):
    """Backtracking search for a valid 3-coloring."""
    colors = [-1] * n
    num_colors = 3
    
    def bt(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(vertex_order):
            return True
        
        v = vertex_order[idx]
        
        # Determine forbidden colors from neighbors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        for c in range(num_colors):
            if c in forbidden:
                continue
            
            colors[v] = c
            
            # Check bicolored constraint incrementally
            if _check_bicolored_incremental(colors, edges, n, v, c):
                if bt(idx + 1):
                    return True
            
            colors[v] = -1
        
        return False
    
    if bt(0):
        return colors[:]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # First try the fast 4-color grid solution
    coloring4 = _four_color_grid(n)
    if _validate_full(coloring4, edges, n):
        best = coloring4
        best_num_colors = 4
    else:
        best = list(range(n))
        best_num_colors = n
    
    # Try to find a 3-coloring with backtracking
    if time.time() < deadline - 5.0:
        # Use degree ordering for better pruning
        vertex_order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        result3 = _backtrack_3color(edges, adj, n, vertex_order, deadline)
        if result3 is not None:
            if _validate_full(result3, edges, n):
                best = result3
                best_num_colors = 3
    
    # Ensure all are Python ints
    best = [int(c) for c in best]
    
    return best

# EVOLVE-BLOCK-END
