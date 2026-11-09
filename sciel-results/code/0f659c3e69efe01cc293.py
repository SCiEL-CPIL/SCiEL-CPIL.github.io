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

def _is_valid_assignment(coloring, n, adj, k):
    """Check all constraints: adjacent differ, and no bicolored cycle for any color pair."""
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    colors_used = sorted(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            ca, cb = colors_used[i], colors_used[j]
            # Collect edges in induced subgraph
            vset = set()
            for v in range(n):
                if coloring[v] == ca or coloring[v] == cb:
                    vset.add(v)
            if not vset:
                continue
            # Union-find on induced subgraph
            parent = {v: v for v in vset}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[ry] = rx
                return False
            for u in vset:
                for w in adj[u]:
                    if w in vset and u < w:
                        if union(u, w):
                            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    adj = _build_adjacency(P['edges'], n)
    
    # Try k=3 first with a smarter search
    # Use a vertex ordering that helps: process in a way that creates tight constraints early
    # Degree ordering
    deg = [len(adj[v]) for v in range(n)]
    
    # Try multiple orderings for robustness
    orderings = []
    
    # Ordering 1: max degree first
    remaining = list(range(n))
    order = []
    for _ in range(n):
        best = max(remaining, key=lambda v: (deg[v], -v))
        order.append(best)
        remaining.remove(best)
    orderings.append(order)
    
    # Ordering 2: min degree first
    remaining = list(range(n))
    order = []
    for _ in range(n):
        best = min(remaining, key=lambda v: (deg[v], v))
        order.append(best)
        remaining.remove(best)
    orderings.append(order)
    
    # Ordering 3: natural order
    orderings.append(list(range(n)))
    
    # Ordering 4: reverse natural
    orderings.append(list(range(n-1, -1, -1)))
    
    for k in range(3, 8):
        if time.time() > deadline - 3.0:
            break
        
        for order in orderings:
            if time.time() > deadline - 2.0:
                break
            
            coloring = [-1] * n
            pos_of_vertex = {v: pos for pos, v in enumerate(order)}
            
            # Precompute neighbors that come earlier in ordering
            neighbor_status = []
            for pos, v in enumerate(order):
                ns = []
                for nb in adj[v]:
                    if nb in pos_of_vertex and pos_of_vertex[nb] < pos:
                        ns.append(nb)
                neighbor_status.append(ns)
            
            # Track which color pairs have been checked for cycles (incremental)
            # For efficiency, we'll do a full check only at the end but prune during search
            
            def backtrack(pos):
                if time.time() > deadline - 1.0:
                    return False
                if pos == n:
                    return True
                
                v = order[pos]
                
                # Get colors used by already-assigned neighbors
                neighbor_colors = set()
                for nb in neighbor_status[pos]:
                    c = coloring[nb]
                    if c != -1:
                        neighbor_colors.add(c)
                
                # Try colors not used by neighbors
                for c in range(k):
                    if c in neighbor_colors:
                        continue
                    coloring[v] = c
                    
                    # Quick bicolored cycle check for pairs (c, ec) where ec is a neighbor color
                    existing_colors = set()
                    for nb in neighbor_status[pos]:
                        cc = coloring[nb]
                        if cc != -1 and cc != c:
                            existing_colors.add(cc)
                    
                    valid = True
                    for ec in existing_colors:
                        # Check if adding v creates a cycle in {c, ec} subgraph
                        # Only check if there's a path between two neighbors of v with color ec
                        # that goes through color c vertices
                        # Simpler: just do a limited DFS from v's neighbors of color ec
                        # to see if they connect through color c vertices
                        
                        # Find all neighbors of v with color ec
                        ec_neighbors = [nb for nb in neighbor_status[pos] if coloring[nb] == ec]
                        if len(ec_neighbors) >= 2:
                            # Check if any two ec-neighbors are connected via a path
                            # that uses only color c or ec vertices
                            # BFS from first ec-neighbor, see if we reach second
                            visited = set()
                            queue = [ec_neighbors[0]]
                            visited.add(ec_neighbors[0])
                            found = False
                            while queue and not found:
                                cur = queue.pop(0)
                                for w in adj[cur]:
                                    if w == v:
                                        continue
                                    if coloring[w] == -1:
                                        continue
                                    if coloring[w] not in (c, ec):
                                        continue
                                    if w not in visited:
                                        visited.add(w)
                                        if w in ec_neighbors[1:]:
                                            found = True
                                            break
                                        queue.append(w)
                            if found:
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
    
    # Fallback: use 50 distinct colors
    coloring = list(range(n))
    return [int(x) for x in coloring]

# EVOLVE-BLOCK-END
