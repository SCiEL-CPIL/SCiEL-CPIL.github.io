"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_bicolor_cycle(coloring, n, adj, color_a, color_b):
    """Check if the subgraph induced by vertices with color_a or color_b has a cycle using union-find."""
    vertices = [v for v in range(n) if coloring[v] == color_a or coloring[v] == color_b]
    if not vertices:
        return False
    vset = set(vertices)
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return True  # cycle detected
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return False
    
    for u in vertices:
        for v in adj[u]:
            if v in vset and u < v:
                if union(u, v):
                    return True
    return False

def _is_valid_assignment(coloring, n, adj, k):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = list(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            if _has_bicolor_cycle(coloring, n, adj, colors_used[i], colors_used[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Precompute a good vertex ordering using greedy degree heuristic
    # Process vertices in order of highest degree first, breaking ties by index
    remaining = list(range(n))
    order = []
    deg = [len(adj[v]) for v in range(n)]
    for _ in range(n):
        # Pick vertex with max degree among remaining
        best = max(remaining, key=lambda v: (deg[v], -v))
        order.append(best)
        remaining.remove(best)
    
    # For each k, try backtracking with the precomputed order
    for k in range(2, 10):
        if time.time() > deadline - 5.0:
            break
        
        coloring = [-1] * n
        # Map: position in order -> vertex index
        # We'll assign colors in order[0], order[1], ...
        
        # Precompute for each position which neighbors are already assigned
        # neighbor_status[pos] = list of (neighbor_vertex, neighbor_pos) where neighbor_pos < pos
        neighbor_status = []
        pos_of_vertex = {}
        for pos, v in enumerate(order):
            pos_of_vertex[v] = pos
            ns = []
            for nb in adj[v]:
                if nb in pos_of_vertex and pos_of_vertex[nb] < pos:
                    ns.append((nb, pos_of_vertex[nb]))
            neighbor_status.append(ns)
        
        def backtrack(pos):
            if time.time() > deadline - 1.0:
                return False
            if pos == n:
                return True
            
            v = order[pos]
            
            # Get colors used by already-assigned neighbors
            neighbor_colors = set()
            for nb, nb_pos in neighbor_status[pos]:
                c = coloring[nb]
                if c != -1:
                    neighbor_colors.add(c)
            
            # Try colors not used by neighbors
            for c in range(k):
                if c in neighbor_colors:
                    continue
                coloring[v] = c
                
                # Check bicolored cycles for all pairs involving c with existing colors
                # Only need to check if adding v creates a cycle
                existing_colors = set()
                for nb, nb_pos in neighbor_status[pos]:
                    cc = coloring[nb]
                    if cc != -1 and cc != c:
                        existing_colors.add(cc)
                
                valid = True
                for ec in existing_colors:
                    # Check if subgraph of {c, ec} has a cycle
                    # Quick check: only look at vertices with color c or ec
                    # Use union-find on the induced subgraph
                    vertices = [u for u in range(n) if coloring[u] == c or coloring[u] == ec]
                    if len(vertices) > 0:
                        vset = set(vertices)
                        uf_parent = {x: x for x in vertices}
                        uf_rank = {x: 0 for x in vertices}
                        
                        def uf_find(x):
                            while uf_parent[x] != x:
                                uf_parent[x] = uf_parent[uf_parent[x]]
                                x = uf_parent[x]
                            return x
                        
                        def uf_union(x, y):
                            rx, ry = uf_find(x), uf_find(y)
                            if rx == ry:
                                return True
                            if uf_rank[rx] < uf_rank[ry]:
                                rx, ry = ry, rx
                            uf_parent[ry] = rx
                            if uf_rank[rx] == uf_rank[ry]:
                                uf_rank[rx] += 1
                            return False
                        
                        has_cycle = False
                        for u in vertices:
                            for w in adj[u]:
                                if w in vset and u < w:
                                    if uf_union(u, w):
                                        has_cycle = True
                                        break
                            if has_cycle:
                                break
                        
                        if has_cycle:
                            valid = False
                            break
                
                if valid:
                    if backtrack(pos + 1):
                        return True
                
                coloring[v] = -1
            
            return False
        
        if backtrack(0):
            if _is_valid_assignment(coloring, n, adj, k):
                return [int(x) for x in coloring]
    
    # Fallback: greedy coloring with many colors
    coloring = list(range(n))
    return [int(x) for x in coloring]

# EVOLVE-BLOCK-END
