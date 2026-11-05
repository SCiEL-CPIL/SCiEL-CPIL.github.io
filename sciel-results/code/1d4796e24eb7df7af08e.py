"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Build edge index: for each edge (u,v) with u<v, store its 1-based arrival index
edge_index = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = i

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    # Build adjacency for each prefix K efficiently
    # We'll scan K from a lower bound upward
    
    # Lower bound: need at least N edges, and each vertex degree >= 2
    # A better lower bound: for each vertex, find when it gets degree 2
    vertex_degree = [0] * N
    vertex_deg2_time = [float('inf')] * N
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        vertex_degree[u] += 1
        vertex_degree[v] += 1
        if vertex_degree[u] == 2:
            vertex_deg2_time[u] = i
        if vertex_degree[v] == 2:
            vertex_deg2_time[v] = i
    
    min_deg2 = max(vertex_deg2_time)
    lo = max(N, min_deg2)
    
    best_cycle = None
    best_K = None
    
    # Scan K from lo upward
    # Build adjacency incrementally
    adj = [set() for _ in range(N)]
    
    for K in range(lo, TOTAL_EDGES + 1):
        if time.time() > deadline:
            break
        
        u, v = EDGE_ORDER[K - 1]
        adj[u].add(v)
        adj[v].add(u)
        
        # Check all vertices have degree >= 2
        if any(len(adj[i]) < 2 for i in range(N)):
            continue
        
        # Try to find Hamiltonian cycle
        cycle = _find_hamilton_cycle(adj, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_K = K
            break  # First K found is the minimum
    
    if best_cycle is not None:
        return best_cycle
    else:
        # Fallback: return identity permutation
        return list(range(N))

def _find_hamilton_cycle(adj, deadline):
    """Find a Hamiltonian cycle in the given adjacency list using backtracking with pruning."""
    N_local = len(adj)
    start = 0
    full_mask = (1 << N_local) - 1
    
    # Precompute neighbor lists sorted by degree (Warnsdorff-like)
    # For each vertex, sort neighbors by their degree (fewer options first)
    neighbor_order = []
    for v in range(N_local):
        neighbors = sorted(adj[v], key=lambda x: len(adj[x]))
        neighbor_order.append(neighbors)
    
    # Use iterative DFS with explicit stack for speed
    # State: (current_vertex, visited_mask, path_tuple)
    # But tuple path is expensive. Use list + backtracking.
    
    path = [0]
    visited = 1  # bitmask
    
    # Recursive backtracking with strong pruning
    def backtrack(v, mask, path):
        if time.time() > deadline:
            return False
        if mask == full_mask:
            return start in adj[v]
        
        # Pruning: check if any unvisited vertex has all its neighbors visited
        # (i.e., it would be isolated in the remaining graph)
        for u in range(N_local):
            if not (mask & (1 << u)):
                # u is unvisited; check if it has any unvisited neighbor or connection to start (if it's the last)
                has_unvisited_neighbor = False
                for w in adj[u]:
                    if not (mask & (1 << w)):
                        has_unvisited_neighbor = True
                        break
                if not has_unvisited_neighbor:
                    # u must be the last vertex to be visited, and it must connect to start
                    # But we can only visit one more vertex (if mask has N-1 bits set)
                    unvisited_count = N_local - bin(mask).count('1')
                    if unvisited_count != 1:
                        return False
                    # u must connect to start
                    if start not in adj[u]:
                        return False
        
        # Try neighbors in order of fewest options first
        for u in neighbor_order[v]:
            if not (mask & (1 << u)):
                path.append(u)
                if backtrack(u, mask | (1 << u), path):
                    return True
                path.pop()
        
        return False
    
    if backtrack(start, visited, path):
        return path
    return None

# EVOLVE-BLOCK-END
