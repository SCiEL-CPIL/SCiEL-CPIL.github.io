"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    adj = [set() for _ in range(P['n'])]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest(coloring, adj):
    """Check that for every pair of colors, the induced subgraph is a forest (acyclic)."""
    n = P['n']
    colors_used = sorted(set(coloring))
    
    # Union-Find for cycle detection
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c1, c2 in combinations(colors_used, 2):
        # Build subgraph induced by vertices with color c1 or c2
        # A cycle exists if we can find one using union-find
        parent = list(range(n))
        has_cycle = False
        
        for u in range(n):
            if coloring[u] not in (c1, c2):
                continue
            for v in adj[u]:
                if v <= u:
                    continue
                if coloring[v] not in (c1, c2):
                    continue
                ru = find(u, parent)
                rv = find(v, parent)
                if ru == rv:
                    has_cycle = True
                    break
                parent[ru] = rv
            if has_cycle:
                break
        if has_cycle:
            return False
    return True


def _count_distinct(coloring):
    return len(set(coloring))


def _is_valid_coloring(coloring, adj, edges):
    """Check if coloring is valid: adjacent vertices differ and bicolored subgraphs are forests."""
    n = P['n']
    # Check adjacent vertices differ
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Check bicolored forest property
    return _check_bicolored_forest(coloring, adj)


def _try_systematic_patterns(k, n, adj, edges, deadline):
    """Try systematic coloring patterns for k colors."""
    # Try patterns based on graph structure
    # The graph appears to be a 5x10 grid with wraparound connections
    
    # Try (row + col * factor) mod k patterns
    # Assuming row-major: vertex i is at (i // 10, i % 10)
    for factor in range(1, k + 1):
        coloring = [((i // 10) + (i % 10) * factor) % k for i in range(n)]
        if _is_valid_coloring(coloring, adj, edges):
            return coloring
    
    # Try (row * factor1 + col * factor2) mod k
    for f1 in range(1, k + 1):
        for f2 in range(1, k + 1):
            coloring = [((i // 10) * f1 + (i % 10) * f2) % k for i in range(n)]
            if _is_valid_coloring(coloring, adj, edges):
                return coloring
    
    # Try linear patterns: (i * p + q) mod k
    for p in range(1, k + 1):
        for q in range(k):
            coloring = [((i * p + q) % k) for i in range(n)]
            if _is_valid_coloring(coloring, adj, edges):
                return coloring
    
    return None


def _backtracking_search(k, n, adj, edges, deadline, rng):
    """Backtracking search for a valid k-coloring."""
    # Order vertices by degree (descending) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    coloring = [-1] * n
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return _is_valid_coloring(coloring, adj, edges)
        
        v = order[idx]
        # Try colors in random order
        colors_to_try = list(range(k))
        rng.shuffle(colors_to_try)
        
        for c in colors_to_try:
            # Check if color c is compatible with already-colored neighbors
            compatible = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            coloring[v] = c
            
            # Check bicolored forest property incrementally
            # For efficiency, only check when we have a complete partial assignment
            # that might create a cycle
            
            if backtrack(idx + 1):
                return True
            
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
        return coloring
    return None


def _local_search(k, n, adj, edges, deadline, rng):
    """Local search: start with a random valid coloring and try to reduce colors."""
    # First, get a valid coloring with k colors using greedy
    coloring = [-1] * n
    order = list(range(n))
    rng.shuffle(order)
    
    for v in order:
        colors_to_try = list(range(k))
        rng.shuffle(colors_to_try)
        for c in colors_to_try:
            compatible = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    compatible = False
                    break
            if compatible:
                coloring[v] = c
                break
    
    if -1 in coloring:
        return None
    
    if not _is_valid_coloring(coloring, adj, edges):
        return None
    
    # Now try to reduce colors by merging
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Pick a random color to try to eliminate
        colors_used = list(set(coloring))
        if len(colors_used) <= 1:
            break
        
        c_to_remove = rng.choice(colors_used)
        target_color = rng.choice([c for c in colors_used if c != c_to_remove])
        
        # Try to recolor all vertices with c_to_remove to target_color
        new_coloring = coloring[:]
        for i in range(n):
            if coloring[i] == c_to_remove:
                new_coloring[i] = target_color
        
        if _is_valid_coloring(new_coloring, adj, edges):
            coloring = new_coloring
            break
    
    return coloring


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Start with a valid coloring: use n colors (one per vertex)
    best_coloring = list(range(n))
    best_count = n
    
    rng = random.Random(42)
    
    # Try small number of colors first
    for k in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 5.0:
            break
        
        # Try systematic patterns first (fast)
        coloring = _try_systematic_patterns(k, n, adj, edges, deadline)
        if coloring is not None:
            cnt = _count_distinct(coloring)
            if cnt < best_count:
                best_coloring = coloring
                best_count = cnt
            if cnt == k:
                break
            continue
        
        # Try backtracking search
        if time.time() < deadline - 10.0:
            coloring = _backtracking_search(k, n, adj, edges, deadline - 5.0, rng)
            if coloring is not None:
                cnt = _count_distinct(coloring)
                if cnt < best_count:
                    best_coloring = coloring
                    best_count = cnt
                if cnt == k:
                    break
                continue
        
        # Try local search
        if time.time() < deadline - 5.0:
            coloring = _local_search(k, n, adj, edges, deadline - 2.0, rng)
            if coloring is not None:
                cnt = _count_distinct(coloring)
                if cnt < best_count:
                    best_coloring = coloring
                    best_count = cnt
                if cnt == k:
                    break
    
    # Ensure we return a list of Python ints
    result = [int(c) for c in best_coloring]
    return result

# EVOLVE-BLOCK-END
