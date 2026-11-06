"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)
START_TIME = time.time()
TIME_LIMIT = 85.0

# Precompute edge arrival index (0-based)
_edge_index = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER):
    if _u > _v:
        _u, _v = _v, _u
    _edge_index[(_u, _v)] = _i

def _get_arrival(u, v):
    """Get 0-based arrival index of edge (u,v)."""
    if u > v:
        u, v = v, u
    return _edge_index[(u, v)]

def _cycle_max_arrival(path):
    """Compute max arrival index (0-based) for a Hamilton cycle given as vertex list."""
    mx = 0
    n = len(path)
    for i in range(n):
        u = path[i]
        v = path[(i + 1) % n]
        a = _get_arrival(u, v)
        if a > mx:
            mx = a
    return mx

def _two_opt_improve(path, deadline):
    """Apply 2-opt local search to minimize max edge arrival index."""
    n = len(path)
    best_max = _cycle_max_arrival(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Try reversing segment [i+1..j]
                # Edges affected: (path[i], path[i+1]) and (path[j], path[j+1])
                # New edges: (path[i], path[j]) and (path[i+1], path[j+1])
                a_old1 = _get_arrival(path[i], path[i + 1])
                a_old2 = _get_arrival(path[j], path[(j + 1) % n])
                a_new1 = _get_arrival(path[i], path[j])
                a_new2 = _get_arrival(path[i + 1], path[(j + 1) % n])
                
                # Compute new max
                # The max of the cycle changes only if old max was one of the removed edges
                # or new max exceeds current max
                old_max = max(a_old1, a_old2)
                new_max = max(a_new1, a_new2)
                
                # If removing these edges doesn't reduce the global max, skip unless it increases
                # We need to check: does the new cycle have a lower max than best_max?
                # Quick check: if both new edges are <= best_max and at least one old edge was == best_max
                # and no other edge has arrival > best_max, then we might improve
                
                # Simpler: just compute if this 2-opt move could help
                # The new max of the entire cycle is max(best_max_without_removed, new_max)
                # This is expensive to compute each time, so use a heuristic:
                # Only accept if new_max < old_max (reduces the bottleneck of the affected edges)
                # AND the overall cycle max doesn't increase
                
                # For efficiency, only try moves where new edges are strictly better
                if new_max < old_max:
                    # Apply the move
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    new_cycle_max = _cycle_max_arrival(path)
                    if new_cycle_max < best_max:
                        best_max = new_cycle_max
                        improved = True
                    else:
                        # Revert
                        path[i+1:j+1] = reversed(path[i+1:j+1])
    return path, best_max

def _greedy_construct(deadline):
    """Construct a Hamilton cycle greedily, preferring edges with low arrival indices."""
    # Build list of edges sorted by arrival index
    edges_sorted = sorted(EDGE_ORDER, key=lambda e: _get_arrival(e[0], e[1]))
    
    # Build adjacency with arrival indices
    adj = [[] for _ in range(N)]
    for u, v in edges_sorted:
        adj[u].append((v, _get_arrival(u, v)))
        adj[v].append((u, _get_arrival(v, u)))
    
    # Greedy: start from vertex 0, always go to unvisited neighbor with lowest arrival edge
    visited = [False] * N
    path = [0]
    visited[0] = True
    
    for _ in range(N - 1):
        current = path[-1]
        best_next = -1
        best_arr = float('inf')
        for v, arr in adj[current]:
            if not visited[v]:
                if arr < best_arr:
                    best_arr = arr
                    best_next = v
        if best_next == -1:
            break
        path.append(best_next)
        visited[best_next] = True
    
    if len(path) != N:
        return None
    
    # Check closing edge exists
    if not any(v == 0 for v, _ in adj[path[-1]]):
        return None
    
    return path

def _build_adjacency(k):
    """Build adjacency masks for edges 1..k (one-based)."""
    masks = [0] * N
    for i in range(k):
        u, v = EDGE_ORDER[i]
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _check_degree(masks):
    """Check each vertex has degree >= 2."""
    for u in range(N):
        if masks[u].bit_count() < 2:
            return False
    return True

def _check_connectivity(masks):
    """Check graph is connected."""
    visited = 1
    stack = [0]
    while stack:
        u = stack.pop()
        neighbors = masks[u] & ~visited
        while neighbors:
            lsb = neighbors & -neighbors
            v = lsb.bit_length() - 1
            visited |= lsb
            stack.append(v)
            neighbors ^= lsb
    return visited == (1 << N) - 1

def _dfs_hamiltonian(masks, start, deadline):
    """
    Find Hamiltonian cycle starting at 'start' using bitmask adjacency.
    Uses Warnsdorff's rule for ordering.
    Returns path (list of N vertices) or None.
    """
    if time.time() > deadline:
        return None
    
    full_mask = (1 << N) - 1
    start_bit = 1 << start
    
    path = [start]
    visited = start_bit
    
    def dfs(u, visited, path):
        if time.time() > deadline:
            return None
        
        if len(path) == N:
            # Check closing edge
            if (masks[u] >> start) & 1:
                return path[:]
            return None
        
        # Get unvisited neighbors
        candidates = masks[u] & ~visited
        
        # If no candidates, dead end
        if candidates == 0:
            return None
        
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        cand_list = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Count unvisited neighbors of v
            deg = (masks[v] & ~visited).bit_count()
            cand_list.append((deg, v))
            temp ^= lsb
        
        cand_list.sort()
        
        for _, v in cand_list:
            new_visited = visited | (1 << v)
            path.append(v)
            res = dfs(v, new_visited, path)
            if res is not None:
                return res
            path.pop()
        
        return None
    
    return dfs(start, visited, path)

def _find_cycle_for_k(k, deadline):
    """Try to find a Hamiltonian cycle using only edges 1..k."""
    if time.time() > deadline:
        return None
    
    masks = _build_adjacency(k)
    
    if not _check_degree(masks):
        return None
    if not _check_connectivity(masks):
        return None
    
    # Try different starting vertices
    for start in range(N):
        if time.time() > deadline:
            break
        path = _dfs_hamiltonian(masks, start, deadline)
        if path is not None:
            return path
    
    return None

def construct():
    deadline = START_TIME + TIME_LIMIT
    
    best_path = None
    best_score = float('inf')
    
    # Strategy 1: Greedy construction + 2-opt improvement
    greedy_path = _greedy_construct(deadline)
    if greedy_path is not None:
        greedy_path, greedy_score = _two_opt_improve(greedy_path.copy(), deadline)
        if greedy_score < best_score:
            best_score = greedy_score
            best_path = greedy_path[:]
    
    # Strategy 2: Binary search for minimal k, then 2-opt
    lo = N
    hi = TOTAL_EDGES
    
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        path = _find_cycle_for_k(mid, deadline)
        if path is not None:
            # Apply 2-opt to improve
            path, score = _two_opt_improve(path, deadline)
            if score < best_score:
                best_score = score
                best_path = path[:]
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Strategy 3: Try multiple random restarts of greedy with different start vertices
    rng = random.Random(614)
    for start in range(N):
        if time.time() > deadline - 5:
            break
        
        # Greedy from different start
        edges_sorted = sorted(EDGE_ORDER, key=lambda e: _get_arrival(e[0], e[1]))
        adj = [[] for _ in range(N)]
        for u, v in edges_sorted:
            adj[u].append((v, _get_arrival(u, v)))
            adj[v].append((u, _get_arrival(v, u)))
        
        visited = [False] * N
        path = [start]
        visited[start] = True
        
        for _ in range(N - 1):
            current = path[-1]
            best_next = -1
            best_arr = float('inf')
            for v, arr in adj[current]:
                if not visited[v]:
                    if arr < best_arr:
                        best_arr = arr
                        best_next = v
            if best_next == -1:
                break
            path.append(best_next)
            visited[best_next] = True
        
        if len(path) == N:
            # Check closing edge
            if any(v == start for v, _ in adj[path[-1]]):
                path, score = _two_opt_improve(path, deadline)
                if score < best_score:
                    best_score = score
                    best_path = path[:]
    
    # Fallback: identity permutation
    if best_path is None:
        return list(range(N))
    
    return best_path

# EVOLVE-BLOCK-END
