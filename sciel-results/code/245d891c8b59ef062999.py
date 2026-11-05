"""Find Hamilton cycle with minimal max edge arrival index in pinned edge order."""
import time
import random
import sys

# EVOLVE-BLOCK-START
_P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = _P['n']
EDGE_ORDER = _P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Precompute edge arrival index (1-based)
_edge_arrival = {}
for i, e in enumerate(EDGE_ORDER):
    a, b = e
    key = (min(a, b), max(a, b))
    _edge_arrival[key] = i + 1

def _cycle_score(cycle):
    """Score = max 1-based arrival index of cycle's edges."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        key = (min(a, b), max(a, b))
        idx = _edge_arrival.get(key, 9999)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _is_connected(mask, adj, start_node):
    """
    Check if the subgraph induced by vertices in 'mask' is connected.
    'adj' is a list of lists of neighbors.
    'start_node' is a node in mask to start BFS from.
    """
    if mask == 0:
        return True
    if not (mask & (1 << start_node)):
        return False
    
    visited = 0
    stack = [start_node]
    visited |= (1 << start_node)
    
    while stack:
        u = stack.pop()
        for v in adj[u]:
            bit = 1 << v
            if (mask & bit) and not (visited & bit):
                visited |= bit
                stack.append(v)
    
    return visited == mask

def _find_hamilton_in_prefix(k, deadline):
    """Find a Hamilton cycle using only first k edges. Returns cycle or None."""
    # Build adjacency from first k edges
    adj = [set() for _ in range(N)]
    for i in range(k):
        a, b = EDGE_ORDER[i]
        adj[a].add(b)
        adj[b].add(a)
    
    # Convert to lists for faster iteration, sorted by degree (least first for pruning)
    adj_lists = [sorted(adj[v], key=lambda x: len(adj[x])) for v in range(N)]
    
    # Check all vertices have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
            
    # Check global connectivity
    full_mask = (1 << N) - 1
    if not _is_connected(full_mask, adj_lists, 0):
        return None

    sys.setrecursionlimit(10000)
    
    best_cycle = None
    deadline_check = 0
    start_time = time.time()
    
    # Fix vertex 0 as start to reduce symmetry
    start = 0
    start_bit = 1 << start
    
    # Try each neighbor of 0 as second vertex
    for second in adj_lists[start]:
        if time.time() > deadline:
            break
        
        second_bit = 1 << second
        initial_mask = start_bit | second_bit
        
        # Check if the remaining vertices (plus start and second) are connected
        # The graph must be connected for the cycle to exist.
        # We check connectivity of the induced subgraph of all vertices.
        # We already did global connectivity.
        # Now in DFS, we need to ensure that the unvisited vertices + current + start remain connected.
        
        path = [start, second]
        
        def dfs(path, mask, current):
            nonlocal best_cycle, deadline_check
            deadline_check += 1
            if deadline_check % 2000 == 0 and time.time() > deadline:
                return None
            
            if len(path) == N:
                # Check if last vertex connects back to start
                if start in adj[current]:
                    return list(path)
                return None
            
            cur = path[-1]
            # Try neighbors not yet visited
            # Heuristic: visit nodes with fewer unvisited neighbors first?
            # Or just sorted by degree.
            
            for nb in adj_lists[cur]:
                nb_bit = 1 << nb
                if mask & nb_bit:
                    continue
                
                new_mask = mask | nb_bit
                new_path = path + [nb]
                
                # Pruning: 
                # The set of vertices in new_mask (visited) plus the unvisited ones must allow a path from nb to start.
                # Specifically, the subgraph induced by (unvisited + {nb, start}) must be connected.
                # unvisited_mask = full_mask ^ new_mask
                # check_mask = unvisited_mask | (1 << nb) | (1 << start)
                # Note: start is already in new_mask, so 1<<start is in new_mask.
                # We need the component containing nb to include start and all unvisited nodes?
                # No, we need the graph formed by (unvisited nodes) + (nb) + (start) to be connected.
                # Actually, standard check: 
                # Let U be unvisited vertices.
                # The vertices U U {nb, start} must induce a connected subgraph.
                # Why? Because we need to go from nb to start visiting all in U.
                # If they are disconnected, we can't form the cycle.
                
                unvisited_mask = full_mask ^ new_mask
                check_mask = unvisited_mask | (1 << nb) | (1 << start)
                
                if not _is_connected(check_mask, adj_lists, nb):
                    continue
                
                result = dfs(new_path, new_mask, nb)
                if result is not None:
                    return result
            
            return None
        
        result = dfs(path, initial_mask, second)
        if result is not None:
            best_cycle = result
            break
    
    return best_cycle

def _find_optimal_cycle(deadline):
    """Binary search + DFS to find cycle with minimal max arrival index."""
    lo, hi = N, NUM_EDGES
    best_cycle = None
    best_score = NUM_EDGES + 1
    
    # First, try to find a good upper bound quickly
    cycle = _find_hamilton_in_prefix(NUM_EDGES, deadline)
    if cycle:
        s = _cycle_score(cycle)
        best_cycle = cycle
        best_score = s
        hi = s
    
    # Binary search for minimal k
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = _find_hamilton_in_prefix(mid, deadline)
        if cycle is not None:
            s = _cycle_score(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    if best_cycle is None:
        cycle = _find_hamilton_in_prefix(NUM_EDGES, deadline)
        if cycle:
            best_cycle = cycle
            best_score = _cycle_score(cycle)
    
    return best_cycle, best_score

def _local_search(cycle, deadline):
    """Try to improve the cycle score via local swaps."""
    if cycle is None:
        return None, 9999
    
    cur_cycle = cycle[:]
    cur_score = _cycle_score(cur_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            for j in range(i + 2, N):
                if i == N - 1 and j == 0: continue # Adjacent in cycle
                
                # Try 2-opt style move: reverse segment or swap
                # Simple swap of vertices i and j
                new_cycle = cur_cycle[:]
                new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
                ns = _cycle_score(new_cycle)
                if ns < cur_score:
                    cur_cycle = new_cycle
                    cur_score = ns
                    improved = True
        # Also try reversing segments
        for i in range(N):
            for j in range(i + 1, N):
                new_cycle = cur_cycle[:]
                # Reverse segment i..j
                segment = new_cycle[i:j+1]
                segment.reverse()
                new_cycle[i:j+1] = segment
                ns = _cycle_score(new_cycle)
                if ns < cur_score:
                    cur_cycle = new_cycle
                    cur_score = ns
                    improved = True
        if time.time() > deadline:
            break
            
    return cur_cycle, cur_score

def construct():
    """Return a permutation of vertices 0..23 describing a Hamilton cycle."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Quick fallback: identity cycle
    fallback = list(range(N))
    fallback_score = _cycle_score(fallback)
    
    best_cycle = None
    best_score = fallback_score
    
    # Strategy 1: Binary search for minimal prefix
    cycle, score = _find_optimal_cycle(deadline)
    if cycle is not None and score < best_score:
        best_cycle = cycle
        best_score = score
    
    # Strategy 2: Local search
    if best_cycle is not None and time.time() < deadline - 5.0:
        lc, ls = _local_search(best_cycle, deadline - 5.0)
        if lc is not None and ls < best_score:
            best_cycle = lc
            best_score = ls
    
    if best_cycle is None:
        return fallback
    return best_cycle
# EVOLVE-BLOCK-END
