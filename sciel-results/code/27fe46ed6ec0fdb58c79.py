"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _is_valid(colors):
    """Check proper coloring and no bicolored cycles using union-find."""
    n = P['n']
    # Check proper coloring
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    # Check no bicolored cycles
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = colors[u], colors[v]
        if cu > cv:
            cu, cv = cv, cu
        pair_edges[(cu, cv)].append((u, v))
    for pair, edges in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False  # cycle found
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True


def _search_2color(deadline):
    """Search for a valid 2-coloring using backtracking with dynamic component tracking."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Order by degree descending
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}
    colors = [-1] * n
    
    # For each color pair (only (0,1) in 2-color case), track connected components
    # We'll use union-find per pair dynamically
    
    # Precompute neighbor lists in order
    neigh = [list(adj[v]) for v in range(n)]
    
    # For 2 colors, only one pair to track: (0, 1)
    # We need to track if assigning colors creates a cycle in the (0,1) subgraph
    
    def backtrack(i):
        if time.time() > deadline:
            return None
        if i == n:
            return list(colors)
        v = order[i]
        
        for c in range(2):
            # Check proper coloring: no neighbor has color c
            ok = True
            for nb in neigh[v]:
                if colors[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check if assigning color c to v creates a bicolored cycle
            # In 2-color case, all edges are in pair (0,1)
            # We need to check if v is already connected to any neighbor with different color
            # through the (0,1) subgraph
            
            # Build a quick check: for each neighbor nb with color 1-c,
            # check if v and nb are already connected in the (0,1) subgraph
            # This requires knowing the current connected components
            
            # Simpler approach: just assign and check at the end, but that's slow
            # Better: maintain union-find incrementally
            
            # For now, use a simpler check: count edges in (0,1) subgraph vs vertices
            # If edges >= vertices in any component, there's a cycle
            # But we need to be more precise
            
            # Let's just assign and do a full check (slower but correct)
            colors[v] = c
            
            # Quick check: if all assigned, verify
            if i == n - 1:
                if _is_valid(colors):
                    return list(colors)
            else:
                # Partial check: just verify no immediate cycle
                # Assign and recurse
                res = backtrack(i + 1)
                if res is not None:
                    return res
            
            colors[v] = -1
        
        return None
    
    return backtrack(0)


def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    # Try 2 colors first with a limited search
    # The graph is bipartite, so 2-coloring exists, but we need no bicolored cycles
    # For 2 colors, the entire graph is one bicolored subgraph, so it must be a forest
    # But the graph has 100 edges and 50 vertices, so it definitely has cycles
    # Therefore, 2 colors is impossible (the bicolored subgraph would be the whole graph with cycles)
    
    # So we need at least 3 colors
    # Try to find a valid 3-coloring
    
    # Use the known 3-coloring pattern: v mod 3
    # Check if it's valid
    colors_3 = [v % 3 for v in range(50)]
    if _is_valid(colors_3):
        return colors_3
    
    # If not, try other 3-coloring patterns
    # Try (v * a + b) % 3 for various a, b
    for a in [1, 2]:
        for b in range(3):
            colors = [(v * a + b) % 3 for v in range(50)]
            if _is_valid(colors):
                return colors
    
    # If none work, do a backtracking search for 3 colors
    n = 50
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos = {v: i for i, v in enumerate(order)}
    colors = [-1] * n
    neigh = [list(adj[v]) for v in range(n)]
    
    # For 3 colors, we need to track 3 pairs: (0,1), (0,2), (1,2)
    # Each pair's subgraph must be a forest
    
    def check_no_cycle_partial():
        """Check if the partial assignment has any bicolored cycle."""
        # For each pair, check if the assigned edges form a forest
        for pair in [(0, 1), (0, 2), (1, 2)]:
            c1, c2 = pair
            parent = list(range(n))
            rank = [0] * n
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            for i, u in enumerate(order):
                cu = colors[u]
                if cu < 0:
                    continue
                for v in neigh[u]:
                    if pos[v] < i:  # Only count each edge once
                        cv = colors[v]
                        if cv < 0 or cv == cu:
                            continue
                        # Check if this edge is in our pair
                        if (cu, cv) == (c1, c2) or (cu, cv) == (c2, c1):
                            ru, rv = find(u), find(v)
                            if ru == rv:
                                return False
                            if rank[ru] < rank[rv]:
                                ru, rv = rv, ru
                            parent[rv] = ru
                            if rank[ru] == rank[rv]:
                                rank[ru] += 1
        return True
    
    def backtrack3(i):
        if time.time() > deadline:
            return None
        if i == n:
            return list(colors)
        v = order[i]
        
        for c in range(3):
            # Check proper coloring
            ok = True
            for nb in neigh[v]:
                if colors[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            colors[v] = c
            
            # Check for bicolored cycles in the partial assignment
            if check_no_cycle_partial():
                res = backtrack3(i + 1)
                if res is not None:
                    return res
            
            colors[v] = -1
        
        return None
    
    res = backtrack3(0)
    if res is not None:
        return res
    
    # Fallback: all distinct colors (always valid)
    return list(range(50))
# EVOLVE-BLOCK-END
