"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles_full(coloring, adj):
    """Full check: verify no bicolored cycles using union-find per color pair."""
    n = P['n']
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair_edges[(cu, cv)].append((u, v))
        elif cv < cu:
            pair_edges[(cv, cu)].append((u, v))
    
    for (c1, c2), edges in pair_edges.items():
        parent = {}
        rank = {}
        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        
        for u, v in edges:
            if not union(u, v):
                return False
    return True

def _get_order(adj):
    """Get vertex ordering: high degree first, then by index for determinism."""
    n = P['n']
    return sorted(range(n), key=lambda x: (-len(adj[x]), x))

def _backtrack_coloring(adj, num_colors, deadline, order=None):
    """Backtracking with incremental bicolored cycle checking."""
    n = P['n']
    coloring = [-1] * n
    
    if order is None:
        order = _get_order(adj)
    
    # Track color usage for symmetry breaking
    color_count = [0] * num_colors
    
    # For each color pair, maintain a DSU to quickly check connectivity
    # We'll use a dict mapping (c1, c2) -> {vertex: parent}
    # But rebuilding DSU each time is expensive. Instead, we do BFS checks.
    
    def would_create_cycle(v, c, adj, coloring):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        neighbor_colors = set()
        for nb in adj[v]:
            if coloring[nb] != -1 and coloring[nb] != c:
                neighbor_colors.add(coloring[nb])
        
        for cn in neighbor_colors:
            # In the subgraph of vertices with color c or cn,
            # check if adding v (color c) creates a cycle.
            # A cycle exists if two of v's cn-colored neighbors are already
            # connected in the {c, cn} subgraph (excluding v).
            
            target_neighbors = [nb for nb in adj[v] if coloring[nb] == cn]
            if len(target_neighbors) < 2:
                # If only one neighbor has color cn, no cycle can form through this pair
                # unless there's a path from v to that neighbor through other {c,cn} vertices
                # Actually, we need to check if v can reach any target neighbor via {c,cn} path
                # of length >= 2 (not the direct edge)
                if len(target_neighbors) == 1:
                    # Check if there's an alternate path
                    target = target_neighbors[0]
                    # BFS from v in {c,cn} subgraph, not using direct edge to target
                    visited = set([v])
                    stack = [v]
                    found = False
                    while stack:
                        cur = stack.pop()
                        for nb2 in adj[cur]:
                            if nb2 == target and cur == v:
                                continue  # skip direct edge
                            if coloring[nb2] not in (c, cn):
                                continue
                            if nb2 in visited:
                                continue
                            if nb2 == target:
                                found = True
                                break
                            visited.add(nb2)
                            stack.append(nb2)
                        if found:
                            break
                    if found:
                        return False
                continue
            
            # Multiple target neighbors: check if any two are connected in {c,cn} subgraph
            # without going through v
            # Build the {c,cn} subgraph excluding v
            # Use BFS/DFS to check connectivity between target neighbors
            
            # Get all vertices with color c or cn
            cn_vertices = set()
            for u in range(n):
                if coloring[u] in (c, cn):
                    cn_vertices.add(u)
            cn_vertices.discard(v)
            
            # Build adjacency in this subgraph
            # For each target neighbor, do BFS to see if we can reach another target neighbor
            for i, t1 in enumerate(target_neighbors):
                if t1 not in cn_vertices:
                    continue
                # BFS from t1
                visited = set([t1])
                stack = [t1]
                while stack:
                    cur = stack.pop()
                    for nb2 in adj[cur]:
                        if nb2 == v:
                            continue
                        if coloring[nb2] not in (c, cn):
                            continue
                        if nb2 in visited:
                            continue
                        if nb2 in target_neighbors:
                            return False  # Cycle: t1 -> ... -> nb2, plus v connects to both
                        visited.add(nb2)
                        stack.append(nb2)
        
        return True
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        
        # Try colors, with symmetry breaking:
        # Only try colors that are either already used or the next unused color
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        
        # Determine which colors to try
        max_used = -1
        for c in range(num_colors):
            if color_count[c] > 0:
                max_used = c
        
        colors_to_try = []
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            # Symmetry breaking: only try color c if c <= max_used+1
            # This ensures we use colors in order 0, 1, 2, ...
            if c <= max_used + 1:
                colors_to_try.append(c)
        
        # Also ensure we don't skip too many
        if not colors_to_try:
            # Must try at least one valid color
            for c in range(num_colors):
                if c not in used_by_neighbors:
                    colors_to_try.append(c)
                    break
        
        for c in colors_to_try:
            coloring[v] = c
            color_count[c] += 1
            
            if would_create_cycle(v, c, adj, coloring):
                result = dfs(idx + 1)
                if result is not None:
                    return result
            
            coloring[v] = -1
            color_count[c] -= 1
        
        return None
    
    return dfs(0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    adj = _build_adj()
    
    # Try with increasing number of colors
    for nc in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, nc, deadline)
        if result is not None:
            # Verify the result
            if _check_bicolored_cycles_full(result, adj):
                return [int(x) for x in result]
    
    # Fallback: use all distinct colors (always valid)
    return [int(i) for i in range(P['n'])]
