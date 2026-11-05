"""Find Hamiltonian cycle minimizing max edge arrival index."""
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Build edge index lookup: (u,v) -> one-based index
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_idx[key] = i + 1


def get_adjacency(threshold):
    """Get adjacency list for edges with index <= threshold."""
    adj = defaultdict(set)
    for i in range(threshold):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_hamiltonian_cycle(threshold, time_limit=55):
    """Find a Hamiltonian cycle using only edges with index <= threshold.
    Returns the cycle as a list of vertices, or None if not found."""
    start_time = time.time()
    adj = get_adjacency(threshold)
    
    # Check basic feasibility: every vertex must have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Use backtracking with symmetry breaking
    # Fix vertex 0 as the start, and its first neighbor as the smallest neighbor
    # to break rotational symmetry
    
    best_cycle = None
    
    def dfs(path, used_edges, current):
        nonlocal best_cycle
        if time.time() - start_time > time_limit:
            return False
        
        if len(path) == N:
            # Check if we can close the cycle: edge from current to 0
            if 0 in adj[current]:
                best_cycle = list(path)
                return True
            return False
        
        # Pruning: check that all unvisited vertices still have enough available edges
        # A vertex in the cycle needs exactly 2 edges total
        # If a vertex is unvisited and has < 2 available neighbors (counting path edges), prune
        
        # For the next vertex, try neighbors in order of their edge index (prefer earlier edges)
        neighbors = list(adj[current])
        # Sort by edge index to prefer earlier edges
        neighbors.sort(key=lambda x: edge_idx[(min(current, x), max(current, x))])
        
        for nxt in neighbors:
            if nxt == 0 and len(path) < N:
                continue  # Can't go back to 0 until the end
            if nxt in path:
                continue
            
            # Pruning: check degree constraints
            # For each unvisited vertex, count how many of its neighbors are in the path
            # or could still be connected
            feasible = True
            for v in range(N):
                if v in path:
                    continue
                # Count available edges for v
                avail = 0
                for nb in adj[v]:
                    if nb in path:
                        avail += 1
                    elif nb == v:
                        continue
                    else:
                        avail += 1  # Could potentially connect
                # Actually, let's do a simpler check:
                # v needs 2 edges in the final cycle
                # Edges to path vertices: at most len(adj[v] & set(path)) can be used
                # Edges to other unvisited: the rest
                path_neighbors = sum(1 for nb in adj[v] if nb in path)
                # v can use at most path_neighbors edges to path
                # and the rest must come from unvisited
                # Total available = path_neighbors + (len(adj[v]) - path_neighbors) = len(adj[v])
                # But we need exactly 2, so we need len(adj[v]) >= 2 (already checked)
                # More useful: if path_neighbors >= 2, v can still work
                # If path_neighbors == 1, v needs exactly 1 more from unvisited
                # If path_neighbors == 0, v needs 2 from unvisited
                pass  # Skip complex pruning for now, rely on basic checks
            
            # Simpler pruning: if adding nxt would make some vertex's degree impossible
            # Check: for each vertex in path (except current and 0), they have exactly 2 edges
            # For current, it has 1 edge so far (from path[-2] to current), needs 1 more
            # For 0, it has 1 edge so far (to path[1]), needs 1 more (the closing edge)
            
            # Check that no vertex in the path (other than endpoints) is over-constrained
            # Actually in a path, internal vertices have 2 edges, endpoints have 1
            # This is automatically satisfied in a simple path
            
            new_path = path + [nxt]
            new_used = used_edges + [(current, nxt)]
            
            if dfs(new_path, new_used, nxt):
                return True
            
            # Backtrack
            del new_path[-1]
        
        return False
    
    # Start: path = [0], try first neighbors
    # Symmetry breaking: fix 0's first neighbor to be the smallest
    first_neighbors = sorted(adj[0])
    for first in first_neighbors:
        if dfs([0, first], [(0, first)], first):
            return best_cycle
    
    return None


def cycle_from_path(path):
    """Convert path [v0, v1, ..., v23] to cycle with closing edge v23->v0."""
    return path


def compute_score(cycle):
    """Compute the score (max edge index) for a given cycle."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        idx = edge_idx[(min(u, v), max(u, v))]
        max_idx = max(max_idx, idx)
    return max_idx


def find_best_cycle():
    """Binary search on threshold to find minimum max edge index."""
    # The minimum possible threshold is N (24 edges needed for a cycle)
    # Maximum is NUM_EDGES (276)
    
    # First, find a feasible solution with full graph
    full_cycle = find_hamiltonian_cycle(NUM_EDGES, time_limit=10)
    if full_cycle is None:
        # Fallback: return identity permutation
        return list(range(N)), N
    
    full_score = compute_score(full_cycle)
    
    # Binary search for minimum threshold
    lo = N  # Need at least N edges
    hi = full_score
    
    best_cycle = full_cycle
    best_score = full_score
    
    while lo <= hi and time.time() < 50:
        mid = (lo + hi) // 2
        cycle = find_hamiltonian_cycle(mid, time_limit=30)
        if cycle is not None:
            score = compute_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_cycle, best_score


# Pre-compute at module level if possible, but use lazy init
_best_cycle = None
_best_score = None


def construct():
    global _best_cycle, _best_score
    if _best_cycle is not None:
        return _best_cycle
    
    start = time.time()
    cycle, score = find_best_cycle()
    _best_cycle = cycle
    _best_score = score
    return cycle
# EVOLVE-BLOCK-END
