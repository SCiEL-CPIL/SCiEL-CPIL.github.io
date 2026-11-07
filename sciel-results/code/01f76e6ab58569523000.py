"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest(coloring, adj, c1, c2):
    """Check if the subgraph induced by colors c1 and c2 is a forest."""
    n = len(coloring)
    # Collect vertices with color c1 or c2
    verts = set()
    for i in range(n):
        if coloring[i] in (c1, c2):
            verts.add(i)
    
    if len(verts) < 2:
        return True
    
    # Union-Find
    parent = {v: v for v in verts}
    rank = {v: 0 for v in verts}
    
    def find(x):
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
    
    # Process all edges between vertices in verts
    for u in verts:
        for v in adj[u]:
            if u < v and v in verts:
                if not union(u, v):
                    return False  # Cycle found
    
    return True

def _is_valid_coloring(coloring, adj):
    """Check if coloring is valid: adjacent differ and no bicolored cycles."""
    n = len(coloring)
    
    # Check adjacent vertices have different colors
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    
    # Check bicolored forest constraint
    colors_used = set(coloring)
    for i in range(len(colors_used)):
        for j in range(i+1, len(colors_used)):
            c1 = list(colors_used)[i]
            c2 = list(colors_used)[j]
            if not _check_bicolored_forest(coloring, adj, c1, c2):
                return False
    
    return True

def _count_distinct(coloring):
    return len(set(coloring))

def _try_greedy_with_patterns(adj, k, deadline):
    """Try to find a valid coloring with k colors using pattern-based approach."""
    n = P['n']
    
    # Try different pattern orderings
    for start_seed in range(100):
        if time.time() > deadline:
            break
        
        # Generate a random-ish ordering based on seed
        import random
        rng = random.Random(start_seed)
        order = list(range(n))
        rng.shuffle(order)
        
        coloring = [-1] * n
        
        # Assign colors in order
        for v in order:
            # Find which colors are forbidden by neighbors
            forbidden = set()
            for nb in adj[v]:
                if coloring[nb] != -1:
                    forbidden.add(coloring[nb])
            
            # Try colors in order
            assigned = False
            for c in range(k):
                if c in forbidden:
                    continue
                
                coloring[v] = c
                
                # Quick check: does this create a bicolored cycle?
                valid = True
                for nb in adj[v]:
                    cn = coloring[nb]
                    if cn == -1 or cn == c:
                        continue
                    # Check if adding this edge creates a cycle in {c, cn}
                    if not _check_bicolored_forest_partial(coloring, adj, c, cn, v, nb):
                        valid = False
                        break
                
                if valid:
                    assigned = True
                    break
                else:
                    coloring[v] = -1
            
            if not assigned:
                break
        
        if all(c != -1 for c in coloring):
            if _is_valid_coloring(coloring, adj):
                return coloring
    
    return None

def _check_bicolored_forest_partial(coloring, adj, c1, c2, new_v, new_nb):
    """Quick check if adding edge (new_v, new_nb) creates a bicolored cycle."""
    n = len(coloring)
    
    # Only consider vertices with color c1 or c2 that are already colored
    verts = set()
    for i in range(n):
        if coloring[i] in (c1, c2) and coloring[i] != -1:
            verts.add(i)
    
    if len(verts) < 2:
        return True
    
    # Union-Find on the existing subgraph
    parent = {v: v for v in verts}
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True
    
    # Process existing edges
    for u in verts:
        for v in adj[u]:
            if u < v and v in verts:
                union(u, v)
    
    # Check if new edge would create a cycle
    if new_v in verts and new_nb in verts:
        if find(new_v) == find(new_nb):
            return False
    
    return True

def _local_search(adj, initial_coloring, k, deadline):
    """Improve coloring via local search."""
    n = P['n']
    coloring = initial_coloring[:]
    best = coloring[:]
    best_count = _count_distinct(coloring)
    
    # Try to reduce number of colors used
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        improved = False
        for v in range(n):
            old_c = coloring[v]
            # Try changing to a color that's already used (to reduce distinct count)
            colors_in_use = set(coloring)
            for c in colors_in_use:
                if c == old_c:
                    continue
                # Check if c is valid for v
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[v] = c
                if _is_valid_coloring(coloring, adj):
                    new_count = _count_distinct(coloring)
                    if new_count < best_count:
                        best_count = new_count
                        best = coloring[:]
                    improved = True
                    break
                else:
                    coloring[v] = old_c
        
        if not improved:
            break
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    adj = _build_adj()
    n = P['n']
    
    # Try increasing number of colors
    for k in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 5:
            break
        
        # Try pattern-based greedy
        result = _try_greedy_with_patterns(adj, k, deadline - 2)
        if result is not None:
            # Try to improve
            improved, count = _local_search(adj, result, k, deadline)
            if count < _count_distinct(result):
                return [int(x) for x in improved]
            else:
                return [int(x) for x in result]
    
    # Fallback: use a simple valid coloring with more colors
    # Use all distinct colors
    return list(range(n))

# EVOLVE-BLOCK-END
