"""Program for the suite task: few colors without any bicolored cycle."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_all_bicolored_forests(coloring, edges, n, K):
    """Check that for every color pair, induced subgraph is a forest."""
    color_to_verts = defaultdict(list)
    for v in range(n):
        color_to_verts[coloring[v]].append(v)
    
    colors = sorted(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            c1, c2 = colors[i], colors[j]
            allowed_set = set(color_to_verts[c1]) | set(color_to_verts[c2])
            if len(allowed_set) < 2:
                continue
            # Union-find
            parent = {v: v for v in allowed_set}
            rank = {v: 0 for v in allowed_set}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return True  # cycle
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                return False
            
            for (u, v) in edges:
                if u in allowed_set and v in allowed_set:
                    if union(u, v):
                        return False
    return True


def _is_valid(coloring, edges, n, K):
    """Check proper coloring and no bicolored cycle."""
    for (u, v) in edges:
        if coloring[u] == coloring[v]:
            return False
    return _check_all_bicolored_forests(coloring, edges, n, K)


def _try_2colors(n, edges, adj):
    """Try to find a valid 2-coloring (bipartite + forest constraint)."""
    # First check if graph is bipartite
    color = [-1] * n
    from collections import deque
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return None  # not bipartite
    # Check bicolored forest constraint
    if _is_valid(color, edges, n, 2):
        return color
    return None


def _try_3colors(n, edges, adj, deadline):
    """Backtracking search for 3-coloring with forward checking on bicolored cycles."""
    
    # Order vertices by degree (highest first)
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    color_of = [-1] * n
    
    # For each color pair, maintain union-find structure
    # pair_key = (min(c1,c2), max(c1,c2))
    # We'll track: for each pair, a dict of parent pointers for vertices in that pair's subgraph
    
    # Instead of maintaining persistent UF, we'll do incremental check
    # When assigning vertex v color c, we need to check if adding v creates a cycle
    # in any pair (c, c') where c' is another color present
    
    # Precompute: for each vertex, which color pairs it could be part of
    # For efficiency, we track components per color pair
    
    # Data structure: pair_components[(c1,c2)] = {vertex: component_id}
    # But component IDs change as we union... Let's use a simpler approach:
    # For each pair, maintain a set of edges that form the forest, and check connectivity
    
    # Actually, let's use a different approach:
    # For each color pair (c1, c2), maintain a list of edges in that subgraph
    # When adding a new edge, check if it creates a cycle using BFS/DFS on that subgraph
    # This is O(E) per check but E is small per pair
    
    # Better: maintain union-find per pair, but we need to support undo
    # For a 50-vertex graph with 3 colors, each pair subgraph has at most 50 vertices
    # and at most ~100 edges total, so per-pair edges are small
    
    # Let's use a recursive backtracking with the following strategy:
    # When assigning color c to vertex v:
    #   For each other color c' in {0,1,2} \ {c}:
    #     Check if adding v (color c) creates a cycle in the (c, c') subgraph
    #     A cycle is created if v has two or more neighbors in the (c,c') subgraph
    #     that are already connected via other (c,c') edges
    
    # For the cycle check: when v gets color c, and we consider pair (c, c'):
    #   v's neighbors that have color c' (or color c) form the new edges
    #   We need to check if any two of v's neighbors in the (c,c') induced subgraph
    #   are already connected (without using v)
    
    # Efficient approach: for each pair (c1,c2), maintain a DSU.
    # When we assign a color, we add edges. When we backtrack, we remove them.
    # Since we're doing DFS backtracking, we can save/restore DSU state.
    
    # Given the small size (50 vertices, 3 colors), let's just do the full check
    # at each step but only for affected pairs.
    
    def check_pair_cycle(c1, c2, v, c):
        """Check if adding vertex v with color c creates a cycle in pair (c1,c2) subgraph.
        Only need to check pairs involving color c."""
        # Find all neighbors of v that have color c1 or c2
        # v has color c, so the pair is (c, c_other)
        c_other = c1 if c2 == c else c2
        
        # Get neighbors of v that have the other color
        other_color_neighbors = []
        for nb in adj[v]:
            if color_of[nb] == c_other:
                other_color_neighbors.append(nb)
        
        if len(other_color_neighbors) < 2:
            return True  # no cycle possible with < 2 connections
        
        # Check if any two of these neighbors are already connected
        # in the (c, c_other) subgraph (excluding v)
        # Build the subgraph of (c, c_other) vertices excluding v
        allowed = set()
        for u in range(n):
            if color_of[u] == c or color_of[u] == c_other:
                if u != v:
                    allowed.add(u)
        
        if len(other_color_neighbors) >= 2:
            # Check connectivity between pairs of other_color_neighbors
            # in the allowed subgraph
            # Use BFS from first neighbor to see if it reaches others
            start = other_color_neighbors[0]
            if start not in allowed:
                return True  # shouldn't happen
            
            visited = set()
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                for nb in adj[node]:
                    if nb in allowed and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            
            for nb in other_color_neighbors[1:]:
                if nb in visited:
                    return False  # cycle detected
        
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = vertex_order[idx]
        
        # Forward checking: determine available colors
        used_by_neighbors = set()
        for nb in adj[v]:
            if color_of[nb] != -1:
                used_by_neighbors.add(color_of[nb])
        
        for c in range(3):
            if c in used_by_neighbors:
                continue
            
            # Check bicolored cycle constraint
            color_of[v] = c
            valid = True
            for c2 in range(3):
                if c2 == c:
                    continue
                if not check_pair_cycle(c, c2, v, c):
                    valid = False
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            color_of[v] = -1
        
        return False
    
    if backtrack(0):
        return color_of
    return None


def _try_4colors(n, edges, adj, deadline):
    """Try 4-coloring with backtracking."""
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    color_of = [-1] * n
    
    def check_pair_cycle(c1, c2, v, c):
        c_other = c1 if c2 == c else c2
        other_color_neighbors = []
        for nb in adj[v]:
            if color_of[nb] == c_other:
                other_color_neighbors.append(nb)
        
        if len(other_color_neighbors) < 2:
            return True
        
        allowed = set()
        for u in range(n):
            if color_of[u] == c or color_of[u] == c_other:
                if u != v:
                    allowed.add(u)
        
        if len(other_color_neighbors) >= 2:
            start = other_color_neighbors[0]
            if start not in allowed:
                return True
            visited = set()
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                for nb in adj[node]:
                    if nb in allowed and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            for nb in other_color_neighbors[1:]:
                if nb in visited:
                    return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = vertex_order[idx]
        used_by_neighbors = set()
        for nb in adj[v]:
            if color_of[nb] != -1:
                used_by_neighbors.add(color_of[nb])
        
        for c in range(4):
            if c in used_by_neighbors:
                continue
            
            color_of[v] = c
            valid = True
            for c2 in range(4):
                if c2 == c:
                    continue
                if not check_pair_cycle(c, c2, v, c):
                    valid = False
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            color_of[v] = -1
        
        return False
    
    if backtrack(0):
        return color_of
    return None


def _try_5colors(n, edges, adj, deadline):
    """Try 5-coloring with backtracking."""
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    color_of = [-1] * n
    
    def check_pair_cycle(c1, c2, v, c):
        c_other = c1 if c2 == c else c2
        other_color_neighbors = []
        for nb in adj[v]:
            if color_of[nb] == c_other:
                other_color_neighbors.append(nb)
        
        if len(other_color_neighbors) < 2:
            return True
        
        allowed = set()
        for u in range(n):
            if color_of[u] == c or color_of[u] == c_other:
                if u != v:
                    allowed.add(u)
        
        if len(other_color_neighbors) >= 2:
            start = other_color_neighbors[0]
            if start not in allowed:
                return True
            visited = set()
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                for nb in adj[node]:
                    if nb in allowed and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            for nb in other_color_neighbors[1:]:
                if nb in visited:
                    return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = vertex_order[idx]
        used_by_neighbors = set()
        for nb in adj[v]:
            if color_of[nb] != -1:
                used_by_neighbors.add(color_of[nb])
        
        for c in range(5):
            if c in used_by_neighbors:
                continue
            
            color_of[v] = c
            valid = True
            for c2 in range(5):
                if c2 == c:
                    continue
                if not check_pair_cycle(c, c2, v, c):
                    valid = False
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            color_of[v] = -1
        
        return False
    
    if backtrack(0):
        return color_of
    return None


def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges, n)
    
    # Try 2 colors first
    result = _try_2colors(n, edges, adj)
    if result is not None:
        return [int(c) for c in result]
    
    # Try 3 colors
    result = _try_3colors(n, edges, adj, deadline)
    if result is not None:
        return [int(c) for c in result]
    
    # Try 4 colors
    if time.time() < deadline:
        result = _try_4colors(n, edges, adj, deadline)
        if result is not None:
            return [int(c) for c in result]
    
    # Try 5 colors
    if time.time() < deadline:
        result = _try_5colors(n, edges, adj, deadline)
        if result is not None:
            return [int(c) for c in result]
    
    # Fallback: greedy with many colors
    coloring = [0] * n
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    for v in vertex_order:
        used = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    
    # Verify and if invalid, use unique colors
    if not _is_valid(coloring, edges, n, 50):
        coloring = list(range(n))
    
    return [int(c) for c in coloring]
# EVOLVE-BLOCK-END
