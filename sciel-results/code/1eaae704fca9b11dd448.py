import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edges = P['edge_order']

# Precompute adjacency for all edges? No, we need subgraphs.
# We will build adjacency masks on the fly for a given K.

def popcount(x):
    return bin(x).count('1')

def find_hamilton_cycle(K, start=0, time_limit=1.0, heuristic='warnsdorff'):
    """
    Returns a list of vertices representing a Hamilton cycle using only the first K edges, or None.
    """
    adj = [0] * n
    for i in range(K):
        u, v = edges[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    for u in range(n):
        if popcount(adj[u]) < 2:
            return None

    full_mask = (1 << n) - 1
    start_mask = 1 << start
    t0 = time.time()
    
    # Iterative DFS with stack for better control
    # State: (current_vertex, path_list, visited_mask, next_candidate_index)
    # We use a stack of (u, path, visited, candidates, candidate_idx)
    
    # Precompute candidate lists per vertex for speed
    # Actually, let's just do iterative with explicit stack
    
    def dfs_iterative():
        # Stack entries: [u, path, visited, cand_list, cand_idx]
        path = [start]
        visited = start_mask
        u = start
        
        # We'll use a recursion-like approach with an explicit stack
        # Each frame: (u, path_tuple, visited, candidates, idx)
        # For memory efficiency, store path as list and backtrack
        
        # Let's use a different approach: iterative with path as list
        path = [start]
        visited = 1 << start
        
        # For each step, we need to try candidates
        # Stack of (u, candidates, next_idx)
        # But we also need the full path for backtracking
        
        # Simpler: use recursion but with a time check
        sys.setrecursionlimit(10000)
        
        def dfs(u, path, visited):
            nonlocal t0
            if time.time() - t0 > time_limit:
                return None
            
            if len(path) == n:
                if adj[u] & start_mask:
                    return path[:]
                return None
            
            cand_mask = adj[u] & ~visited
            if cand_mask == 0:
                return None
            
            candidates = []
            m = cand_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                
                if len(path) == n - 1:
                    if adj[v] & start_mask:
                        candidates.append((0, v))
                    else:
                        candidates.append((1, v))
                elif heuristic == 'warnsdorff':
                    score = popcount(adj[v] & ~visited)
                    candidates.append((score, v))
                else:
                    candidates.append((0, v))
            
            candidates.sort()
            
            for _, v in candidates:
                # Pruning: check if any unvisited vertex is stuck
                next_unvisited = ~visited & ~(1 << v) & full_mask
                is_valid = True
                
                m_w = next_unvisited
                while m_w:
                    lsb_w = m_w & -m_w
                    w = lsb_w.bit_length() - 1
                    m_w ^= lsb_w
                    
                    avail = adj[w] & (next_unvisited | (1 << v) | start_mask)
                    deg_avail = popcount(avail)
                    
                    if deg_avail < 2:
                        if not ((adj[w] & start_mask) and deg_avail == 1):
                            is_valid = False
                            break
                    # Also check: if w has 0 neighbors in next_unvisited and 
                    # w != v and not adjacent to start, it's stuck unless it's the last
                    # Actually covered above
                    
                if not is_valid:
                    continue
                
                # Additional pruning: articulation point check
                # If removing v from the unvisited subgraph disconnects it, prune
                # (unless the disconnected component can be the "end" connecting to start)
                # This is expensive; skip for now
                
                path.append(v)
                res = dfs(v, path, visited | (1 << v))
                path.pop()
                if res:
                    return res
            
            return None
        
        return dfs(start, [start], 1 << start)
    
    return dfs_iterative()

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Binary search for K
    lo = 24
    hi = len(edges)
    best_cycle = None
    best_K = hi
    
    # First check if full graph works (sanity)
    cycle = find_hamilton_cycle(hi, start=0, time_limit=5.0)
    if cycle:
        best_cycle = cycle
        best_K = hi
    
    while lo < hi:
        if time.time() > deadline:
            break
        
        mid = (lo + hi) // 2
        
        # Try multiple starts and heuristics for better success rate
        cycle = None
        remaining = deadline - time.time()
        if remaining > 1.0:
            # Try with start=0, warnsdorff
            cycle = find_hamilton_cycle(mid, start=0, time_limit=min(remaining * 0.3, 5.0), heuristic='warnsdorff')
        
        if cycle is None and time.time() < deadline:
            # Try different start vertex
            remaining = deadline - time.time()
            if remaining > 1.0:
                cycle = find_hamilton_cycle(mid, start=1, time_limit=min(remaining * 0.3, 5.0), heuristic='warnsdorff')
        
        if cycle is None and time.time() < deadline:
            # Try another start
            remaining = deadline - time.time()
            if remaining > 1.0:
                cycle = find_hamilton_cycle(mid, start=2, time_limit=min(remaining * 0.3, 5.0), heuristic='warnsdorff')
        
        if cycle:
            best_cycle = cycle
            best_K = mid
            hi = mid
        else:
            lo = mid + 1
    
    # After binary search, try to verify/improve at the boundary
    if time.time() < deadline:
        # Try a few more values around the found boundary
        for K_try in range(max(24, best_K - 2), best_K + 1):
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 1.0:
                break
            for s in [0, 1, 2, 3]:
                if time.time() > deadline:
                    break
                cycle = find_hamilton_cycle(K_try, start=s, time_limit=min(remaining * 0.2, 3.0), heuristic='warnsdorff')
                if cycle:
                    best_cycle = cycle
                    best_K = K_try
                    break
    
    if best_cycle:
        return best_cycle
    else:
        # Fallback: return identity (should not happen)
        return list(range(n))

# EVOLVE-BLOCK-END
