import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
EDGE_POS = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    EDGE_POS[(min(u,v), max(u,v))] = i + 1  # 1-based

def edge_index(u, v):
    return EDGE_POS[(min(u,v), max(u,v))]

def cycle_max_edge(cycle):
    """Compute max edge index for a cycle."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i+1) % len(cycle)]
        idx = edge_index(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def find_hamilton_cycle_bitmask(adj_bitmasks, n, time_limit=10, start_time=None, start_vertex=0):
    """Find Hamilton cycle using bitmask backtracking with Warnsdorff heuristic."""
    if start_time is None:
        start_time = time.time()
    
    # Check degrees
    for v in range(n):
        if bin(adj_bitmasks[v]).count('1') < 2:
            return None
    
    visited_mask = 1 << start_vertex
    path = [start_vertex]
    
    def get_unvisited_neighbors(cur, visited):
        """Get unvisited neighbors of cur, sorted by Warnsdorff (fewest unvisited neighbors first)."""
        candidates = adj_bitmasks[cur] & ~visited
        if candidates == 0:
            return []
        nbrs = []
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            nbrs.append(v)
            m ^= lsb
        nbrs.sort(key=lambda v: bin(adj_bitmasks[v] & ~visited).count('1'))
        return nbrs
    
    def backtrack():
        if time.time() - start_time > time_limit:
            return None
        if len(path) == n:
            if adj_bitmasks[path[-1]] & (1 << start_vertex):
                return path[:]
            return None
        
        cur = path[-1]
        nbrs = get_unvisited_neighbors(cur, visited_mask)
        if not nbrs:
            return None
        
        for v in nbrs:
            visited_mask_new = visited_mask | (1 << v)
            path.append(v)
            result = backtrack()
            if result is not None:
                return result
            path.pop()
        
        return None
    
    return backtrack()

def two_opt_refine(cycle, time_limit=20, start_time=None):
    """Apply 2-opt moves to reduce max edge index."""
    if start_time is None:
        start_time = time.time()
    
    n = len(cycle)
    best_cycle = cycle[:]
    best_max = cycle_max_edge(cycle)
    
    # Precompute edge indices for all pairs
    # cycle_edges[i] = edge_index between cycle[i] and cycle[(i+1)%n]
    
    while time.time() - start_time < start_time + time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > start_time + time_limit:
                break
            for j in range(i+2, n):
                if j == (i-1) % n:
                    continue
                a = best_cycle[i]
                b = best_cycle[(i+1) % n]
                c = best_cycle[j]
                d = best_cycle[(j+1) % n]
                
                # Try 2-opt: replace edges (a,b) and (c,d) with (a,c) and (b,d)
                # This reverses the segment from b to c
                e_ac = edge_index(a, c)
                e_bd = edge_index(b, d)
                
                # Only worth trying if both new edges are < current best_max
                if e_ac < best_max and e_bd < best_max:
                    # Construct new cycle
                    # Original: a, b, ..., c, d
                    # New: a, c, (reversed b...c), d
                    # Segment from b to c (inclusive) is best_cycle[i+1 : j+1]
                    seg = best_cycle[i+1:j+1]
                    seg.reverse()
                    new_cycle = best_cycle[:i+1] + seg + best_cycle[j+1:]
                    
                    # Verify
                    valid = True
                    for k in range(n):
                        u = new_cycle[k]
                        v = new_cycle[(k+1) % n]
                        if (u,v) not in EDGE_POS and (v,u) not in EDGE_POS:
                            valid = False
                            break
                    if valid:
                        nm = cycle_max_edge(new_cycle)
                        if nm < best_max:
                            best_max = nm
                            best_cycle = new_cycle
                            improved = True
                            break
            if improved:
                break
        if not improved:
            break
    
    return best_cycle, best_max

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Build full adjacency bitmasks
    full_adj = [0] * N
    for u, v in EDGE_ORDER:
        full_adj[u] |= (1 << v)
        full_adj[v] |= (1 << u)
    
    def build_adj_bitmasks(T):
        adj = [0] * N
        for i in range(T):
            u, v = EDGE_ORDER[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    lo, hi = N, 276
    best_cycle = None
    best_T = 277
    
    # Find a cycle in the full graph
    adj_full = build_adj_bitmasks(276)
    for sv in range(N):
        if time.time() - start_time > deadline:
            break
        cycle = find_hamilton_cycle_bitmask(adj_full, N, time_limit=10, start_time=time.time(), start_vertex=sv)
        if cycle is not None:
            max_idx = cycle_max_edge(cycle)
            best_T = max_idx
            best_cycle = cycle
            break
    
    if time.time() - start_time > deadline:
        return best_cycle if best_cycle else list(range(N))
    
    # Binary search for smaller T
    while lo < hi and time.time() - start_time < deadline:
        mid = (lo + hi) // 2
        adj = build_adj_bitmasks(mid)
        
        valid = True
        for v in range(N):
            if bin(adj[v]).count('1') < 2:
                valid = False
                break
        
        if valid:
            found = False
            for sv in range(0, N, 2):
                if time.time() - start_time > deadline:
                    break
                cycle = find_hamilton_cycle_bitmask(adj, N, time_limit=5, start_time=time.time(), start_vertex=sv)
                if cycle is not None:
                    max_idx = cycle_max_edge(cycle)
                    if max_idx <= mid:
                        hi = mid
                        best_T = max_idx
                        best_cycle = cycle
                        found = True
                        break
            if not found:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Local search refinement
    if best_cycle is not None and time.time() - start_time < deadline - 5:
        refined_cycle, refined_max = two_opt_refine(
            best_cycle, 
            time_limit=min(30, deadline - 5 - (time.time() - start_time)),
            start_time=time.time()
        )
        if refined_max < best_T:
            best_T = refined_max
            best_cycle = refined_cycle
    
    # Try to find even better cycles
    if best_cycle is not None and time.time() - start_time < deadline - 2:
        target_T = best_T - 1
        if target_T >= N:
            adj = build_adj_bitmasks(target_T)
            valid = True
            for v in range(N):
                if bin(adj[v]).count('1') < 2:
                    valid = False
                    break
            if valid:
                for sv in range(N):
                    if time.time() - start_time > deadline - 2:
                        break
                    cycle = find_hamilton_cycle_bitmask(adj, N, time_limit=3, start_time=time.time(), start_vertex=sv)
                    if cycle is not None:
                        max_idx = cycle_max_edge(cycle)
                        if max_idx < best_T:
                            best_T = max_idx
                            best_cycle = cycle
                            break
    
    if best_cycle is None:
        return list(range(N))
    
    return best_cycle

# EVOLVE-BLOCK-END
