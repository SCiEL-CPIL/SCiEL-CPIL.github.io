"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(colors, adj, n):
    """Check bicolored forest constraint using union-find for each color pair."""
    colors_present = sorted(set(colors))
    if len(colors_present) < 2:
        return True
    
    for i in range(len(colors_present)):
        for j in range(i + 1, len(colors_present)):
            c1 = colors_present[i]
            c2 = colors_present[j]
            
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
            
            for u in range(n):
                if colors[u] == c1 or colors[u] == c2:
                    pass  # already in parent
            
            for u, v in P['edges']:
                if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                    if not union(u, v):
                        return False
    
    return True

def _is_valid_coloring(colors, adj, n):
    """Check if coloring is valid: proper + bicolored forest."""
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    if not _check_bicolor_forest(colors, adj, n):
        return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _backtrack_with_pruning(adj, n, max_colors, deadline):
    """
    Backtracking with strong pruning:
    - Proper coloring constraint
    - Bicolored cycle detection via union-find
    """
    colors = [-1] * n
    
    # Precompute for each vertex, which colors are forbidden by neighbors
    # Use a global union-find per color pair to track connectivity
    
    # For efficiency, track for each (c1, c2) pair, the union-find state
    # We'll use a simpler approach: when assigning a vertex, check if it creates
    # a bicolored cycle
    
    def find_in_pair(c1, c2, x, uf):
        """Find root in union-find for color pair (c1, c2)."""
        while uf[c1][c2][x] != x:
            uf[c1][c2][x] = uf[c1][c2][uf[c1][c2][x]]
            x = uf[c1][c2][x]
        return x
    
    def union_in_pair(c1, c2, x, y, uf, rank):
        """Union in union-find for color pair (c1, c2). Returns False if cycle."""
        rx = find_in_pair(c1, c2, x, uf)
        ry = find_in_pair(c1, c2, y, uf)
        if rx == ry:
            return False, None, None
        if rank[c1][c2][rx] < rank[c1][c2][ry]:
            rx, ry = ry, rx
        uf[c1][c2][ry] = rx
        if rank[c1][c2][rx] == rank[c1][c2][ry]:
            rank[c1][c2][rx] += 1
        return True, rx, ry
    
    # Initialize union-find structures for each color pair
    # uf[c1][c2][v] = parent of v in the union-find for colors c1, c2
    # We only care about vertices that have color c1 or c2
    
    max_c = max_colors
    uf = [[[i for i in range(n)] for c2 in range(max_c)] for c1 in range(max_c)]
    rank = [[[0] * n for c2 in range(max_c)] for c1 in range(max_c)]
    
    # Count of vertices with each color
    color_count = [0] * max_c
    
    def is_feasible(v, c):
        """Check if vertex v can be colored with color c."""
        # Proper coloring: no neighbor has color c
        for u in adj[v]:
            if colors[u] == c:
                return False, None, None, None
        
        # Bicolored cycle check: for each neighbor color c2 != c,
        # check if adding v creates a cycle in {c, c2} subgraph
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] != -1:
                neighbor_colors.add(colors[u])
        
        for c2 in neighbor_colors:
            if c2 == c:
                continue
            
            # Find neighbors of v with color c2
            neighbors_in_c2 = [u for u in adj[v] if colors[u] == c2]
            if len(neighbors_in_c2) < 2:
                continue
            
            # Check if any two neighbors in c2 are already connected
            # through the {c, c2} subgraph (excluding v)
            # If they are, adding v would create a cycle
            first = neighbors_in_c2[0]
            for other in neighbors_in_c2[1:]:
                if find_in_pair(c, c2, first, uf) == find_in_pair(c, c2, other, uf):
                    return False, None, None, None
        
        return True, c, None, None
    
    def assign(v, c):
        """Assign color c to vertex v. Returns list of (c1, c2, x, y) for undo."""
        undo = []
        colors[v] = c
        color_count[c] += 1
        
        # For each neighbor u, if u has color c2, union v and u in pair (c, c2)
        for u in adj[v]:
            if colors[u] != -1 and colors[u] != c:
                c2 = colors[u]
                success, rx, ry = union_in_pair(c, c2, v, u, uf, rank)
                if not success:
                    # This shouldn't happen if is_feasible was checked
                    # But undo and return failure
                    for (uc1, uc2, ux, uy) in undo:
                        # This is complex; in practice we should prevent this
                        pass
                    return None
                undo.append((c, c2, v, u))
        
        return undo
    
    def unassign(v, c, undo):
        """Undo assignment of color c to vertex v."""
        for (c1, c2, x, y) in undo:
            # We need to split the union, which is hard
            # Instead, we'll rebuild or use a different approach
            pass
        colors[v] = -1
        color_count[c] -= 1
    
    # Simpler approach: use BFS-based cycle detection for feasibility
    def is_feasible_simple(v, c):
        """Check if vertex v can be colored with color c using BFS."""
        # Proper coloring
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Bicolored cycle check
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] != -1:
                neighbor_colors.add(colors[u])
        
        for c2 in neighbor_colors:
            if c2 == c:
                continue
            
            neighbors_in_c2 = [u for u in adj[v] if colors[u] == c2]
            if len(neighbors_in_c2) < 2:
                continue
            
            # BFS from neighbors_in_c2[0] through {c, c2} vertices
            # to see if we can reach any other neighbor_in_c2
            start = neighbors_in_c2[0]
            target_set = set(neighbors_in_c2[1:])
            
            visited = set()
            queue = deque([start])
            visited.add(start)
            
            while queue:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    if nxt == v:
                        continue
                    if colors[nxt] != c and colors[nxt] != c2:
                        continue
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    if nxt in target_set:
                        return False
                    queue.append(nxt)
        
        return True
    
    # Vertex ordering: use a heuristic (e.g., most constrained first)
    # For simplicity, use natural order or BFS order
    order = list(range(n))
    
    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors 0 to max_colors-1
        for c in range(max_colors):
            if is_feasible_simple(v, c):
                colors[v] = c
                if bt(idx + 1):
                    return True
                colors[v] = -1
        return False
    
    if bt(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin
    n = P['n']
    adj = _build_adjacency()
    
    best_result = None
    best_score = n + 1
    
    # Try increasing number of colors
    for k in range(2, 12):
        if time.time() > deadline - 5.0:
            break
        
        result = _backtrack_with_pruning(adj, n, k, deadline)
        if result is not None:
            if _is_valid_coloring(result, adj, n):
                distinct = _count_distinct(result)
                if distinct < best_score:
                    best_score = distinct
                    best_result = result[:]
                break  # Found valid coloring with k colors, try smaller next
    
    # Fallback: use all distinct colors
    if best_result is None:
        best_result = list(range(n))
        best_score = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
