"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _has_bicolored_cycle_full(coloring, edges):
    """Full check: any pair of colors induces a cycle?"""
    n = len(coloring)
    color_to_verts = defaultdict(list)
    for i, c in enumerate(coloring):
        color_to_verts[c].append(i)
    
    colors = list(color_to_verts.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            ci, cj = colors[i], colors[j]
            verts = set(color_to_verts[ci]) | set(color_to_verts[cj])
            if len(verts) < 3:
                continue
            
            parent = {v: v for v in verts}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            for u, v in edges:
                if coloring[u] in (ci, cj) and coloring[v] in (ci, cj):
                    if union(u, v):
                        return True
    return False

def _is_valid_full(coloring, edges):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle_full(coloring, edges)

def _try_periodic_pattern(n, edges, period, k):
    """Try all k-colorings of the form color[i] = f(i % period) for all f: Z_p -> Z_k.
    Returns the best coloring found or None."""
    n = 50
    best = None
    best_count = n + 1
    
    # Enumerate all functions f: {0,...,period-1} -> {0,...,k-1}
    # That's k^period possibilities. For period=5, k=3: 243 patterns.
    # For period=7, k=3: 2187 patterns. Manageable.
    
    import itertools
    for pattern in itertools.product(range(k), repeat=period):
        coloring = [pattern[i % period] for i in range(n)]
        
        # Quick proper coloring check
        valid = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                valid = False
                break
        if not valid:
            continue
        
        # Check bicolored cycles
        if _has_bicolored_cycle_full(coloring, edges):
            continue
        
        cnt = len(set(coloring))
        if cnt < best_count:
            best_count = cnt
            best = list(coloring)
            if cnt == 3:
                break
    
    return best, best_count

def _backtrack_k_coloring(n, edges, k, deadline):
    """Backtracking search for a valid k-coloring."""
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (all degree 4 here, so just natural order)
    order = list(range(n))
    
    color = [-1] * n
    # For incremental bicolored cycle detection, we track DSU per color pair
    # But that's complex. Instead, we do a lighter check at each step.
    
    # Precompute: for each vertex, which colors are forbidden by neighbors
    # We'll use forward checking
    
    def backtrack(v_idx):
        if time.time() > deadline:
            return False
        if v_idx == n:
            # Final full check
            return not _has_bicolored_cycle_full(color, edges)
        
        v = order[v_idx]
        # Determine allowed colors
        allowed = set(range(k))
        for nb in adj[v]:
            if color[nb] >= 0:
                allowed.discard(color[nb])
        
        if not allowed:
            return False
        
        # Try each allowed color
        for c in sorted(allowed):
            color[v] = c
            
            # Forward check: for each uncolored neighbor, verify it still has options
            ok = True
            for nb in adj[v]:
                if color[nb] < 0:
                    nb_allowed = set(range(k))
                    for nnb in adj[nb]:
                        if color[nnb] >= 0:
                            nb_allowed.discard(color[nnb])
                    if not nb_allowed:
                        ok = False
                        break
            
            if ok and backtrack(v_idx + 1):
                return True
            
            color[v] = -1
        
        return False
    
    if backtrack(0):
        return color[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Fallback: 50 distinct colors
    best = list(range(n))
    best_count = n
    
    # Phase 1: Try periodic patterns for k=3 with small periods
    # The graph is circulant C(50, {1,7}), so period must relate to 50
    # Try periods that divide 50 or are small
    for period in [5, 10, 25, 50]:
        if time.time() > deadline:
            break
        result, cnt = _try_periodic_pattern(n, edges, period, 3)
        if result is not None and cnt < best_count:
            best = result
            best_count = cnt
            if best_count == 3:
                return [int(x) for x in best]
    
    # Phase 2: Try k=4 periodic patterns
    for period in [5, 10, 25]:
        if time.time() > deadline:
            break
        result, cnt = _try_periodic_pattern(n, edges, period, 4)
        if result is not None and cnt < best_count:
            best = result
            best_count = cnt
            if best_count <= 4:
                break
    
    # Phase 3: Backtracking for k=3 if not found yet
    if best_count > 3 and time.time() < deadline:
        result = _backtrack_k_coloring(n, edges, 3, deadline)
        if result is not None:
            best = result
            best_count = 3
    
    # Phase 4: Backtracking for k=4 if still not found
    if best_count > 4 and time.time() < deadline:
        result = _backtrack_k_coloring(n, edges, 4, deadline)
        if result is not None:
            best = result
            best_count = 4
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
