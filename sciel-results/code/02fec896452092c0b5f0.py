"""Find minimal Hamilton cycle using constraint propagation and iterative deepening."""
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Build edge arrival index map
_edge_arrival = {}
for idx, (u, v) in enumerate(EDGE_ORDER):
    if u > v:
        u, v = v, u
    _edge_arrival[(u, v)] = idx + 1

def _build_adj(k):
    """Build adjacency sets for subgraph with edges 1..k."""
    adj = [set() for _ in range(N)]
    for i in range(k):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamilton_cycle(adj, deadline):
    """
    Find Hamilton cycle using backtracking with strong pruning.
    Returns list of vertices in cycle order, or None.
    """
    # Quick degree check
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Use bitmasks for speed
    adj_mask = [0] * N
    for i in range(N):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    all_visited_mask = (1 << N) - 1
    
    def _backtrack(path, visited_mask, prev):
        if time.time() > deadline:
            return None
        
        if len(path) == N:
            # Check if we can close the cycle
            if prev in adj[0]:
                return list(path)
            return None
        
        # Pruning: check that all unvisited vertices have at least 2 "available" connections
        # An unvisited vertex v needs to connect to two other vertices in the cycle
        # Those can be: unvisited neighbors, or the endpoints of the path (0 and prev) if adjacent
        unvisited = all_visited_mask ^ visited_mask
        if unvisited:
            # Count unvisited vertices with insufficient connections
            for v in range(N):
                if not (visited_mask & (1 << v)):
                    # Available neighbors: unvisited neighbors + (0 if adjacent and path length > 1) + (prev if adjacent)
                    avail_count = 0
                    # Unvisited neighbors
                    unvisited_neighbors = adj_mask[v] & unvisited
                    avail_count = bin(unvisited_neighbors).count('1')
                    # Can connect to 0 (start of cycle) - but only if we're not at the last step
                    # Actually, 0 is already visited, so it can only be used for the final connection
                    # For intermediate steps, v can connect to prev (if adjacent)
                    if prev in adj[v]:
                        avail_count += 1
                    # v can also connect to 0 for closing, but that's only at the end
                    # For now, we need at least 2 total connections in the final cycle
                    # If avail_count < 2, it might still work if it can connect to 0 at the end
                    # But if it can't connect to prev AND has < 2 unvisited neighbors, it's stuck
                    if avail_count < 2:
                        # Check if it can potentially connect to 0 (for closing)
                        if 0 in adj[v]:
                            # It needs at least 1 unvisited neighbor or connection to prev
                            # plus the connection to 0
                            if avail_count < 1:
                                return None
                        else:
                            return None
        
        # Try neighbors of prev, ordered by degree (least first for more pruning)
        candidates = []
        for nb in adj[prev]:
            if not (visited_mask & (1 << nb)):
                candidates.append(nb)
        
        # Sort by degree (fewer options first)
        candidates.sort(key=lambda x: len(adj[x]))
        
        for nb in candidates:
            new_mask = visited_mask | (1 << nb)
            result = _backtrack(path + [nb], new_mask, nb)
            if result is not None:
                return result
        
        return None
    
    # Fix vertex 0 as start. Try each possible first neighbor.
    # To break symmetry, we only need to try one direction, but let's try all for safety.
    for first_nb in sorted(adj[0], key=lambda x: len(adj[x])):
        result = _backtrack([0, first_nb], (1 << 0) | (1 << first_nb), first_nb)
        if result is not None:
            return result
    
    return None

def _cycle_score(cycle):
    """Compute the max arrival index of edges in the cycle."""
    max_idx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        if u > v:
            u, v = v, u
        idx = _edge_arrival.get((u, v), NUM_EDGES)
        max_idx = max(max_idx, idx)
    return max_idx

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Linear scan from low K upward
    # The answer is likely small, so this is efficient
    for k in range(N, NUM_EDGES + 1):  # Need at least N edges for a cycle
        if time.time() > deadline - 5:
            break
        
        adj = _build_adj(k)
        
        # Quick check: all vertices must have degree >= 2
        feasible = True
        for i in range(N):
            if len(adj[i]) < 2:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Try to find Hamilton cycle
        cycle = _find_hamilton_cycle(adj, deadline - 2)
        if cycle is not None:
            return cycle
    
    # Fallback: return identity cycle
    return list(range(N))
# EVOLVE-BLOCK-END
