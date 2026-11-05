"""Find minimal Hamilton cycle in edge process using binary search + backtracking."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Build edge arrival index map: edge (u,v) with u<v -> arrival index (1-based)
_edge_arrival = {}
for idx, (u, v) in enumerate(EDGE_ORDER):
    if u > v:
        u, v = v, u
    _edge_arrival[(u, v)] = idx + 1

def _get_adjacency(k):
    """Build adjacency list for subgraph with edges having arrival <= k."""
    adj = [set() for _ in range(N)]
    for idx, (u, v) in enumerate(EDGE_ORDER):
        if idx + 1 > k:
            break
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_hamilton_cycle(adj):
    """Check if graph with given adjacency has a Hamilton cycle using backtracking with pruning."""
    # Quick check: every vertex must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return False
    
    # Try to find a Hamilton cycle starting from vertex 0
    # Fix vertex 0 as start, and to avoid symmetry, fix the first neighbor
    # Use backtracking with degree pruning
    
    visited = [False] * N
    path = [0]
    visited[0] = True
    # Track remaining degree requirements
    # For a Hamilton cycle, each vertex needs exactly 2 edges in the cycle
    
    def backtrack(pos, prev):
        if pos == N:
            # Check if we can close the cycle: last vertex must connect to vertex 0
            if 0 in adj[path[-1]]:
                return True
            return False
        
        # Pruning: check if any unvisited vertex has degree < 2 in remaining graph
        # This is expensive, so do it selectively
        if pos % 4 == 0:
            for v in range(N):
                if not visited[v]:
                    # Count available neighbors (unvisited + endpoints 0 and current last)
                    avail = 0
                    for nb in adj[v]:
                        if not visited[nb]:
                            avail += 1
                        elif nb == 0 and pos == N - 1:
                            avail += 1
                    # Also count if v connects to current last (which is being left)
                    # Actually, for unvisited vertices, they can connect to unvisited or to 0 (if closing)
                    # or to the last vertex in path (if it's the final step)
                    if avail < 2:
                        # Special case: if v is the last to be visited, it only needs 1 more (to connect to 0)
                        if pos == N - 1:
                            # v needs to connect to path[-1] and to 0
                            if not (v in adj[path[-1]] and 0 in adj[v]):
                                return False
                        else:
                            return False
        
        for nb in adj[prev]:
            if not visited[nb]:
                visited[nb] = True
                path.append(nb)
                if backtrack(pos + 1, nb):
                    return True
                path.pop()
                visited[nb] = False
        
        return False
    
    # To break symmetry, try neighbors of 0 in order
    # We need a cycle 0 -> v1 -> v2 -> ... -> v23 -> 0
    # Fix 0, try each possible first step
    for first_nb in sorted(adj[0]):
        # We'll do a full backtracking from here
        visited = [False] * N
        visited[0] = True
        path = [0, first_nb]
        visited[first_nb] = True
        
        def backtrack2(pos):
            if pos == N:
                return 0 in adj[path[-1]]
            
            # Pruning: check unvisited vertices
            if pos % 3 == 0:
                for v in range(N):
                    if not visited[v]:
                        cnt = 0
                        for nb in adj[v]:
                            if not visited[nb]:
                                cnt += 1
                            elif nb == 0:
                                # Only counts if we're at the last position
                                if pos == N - 1:
                                    cnt += 1
                        if cnt < 2:
                            if pos == N - 1:
                                # Last vertex needs connection to path[-1] and 0
                                if not (v in adj[path[-1]] and 0 in adj[v]):
                                    return False
                            else:
                                return False
            
            for nb in adj[path[-1]]:
                if not visited[nb]:
                    visited[nb] = True
                    path.append(nb)
                    if backtrack2(pos + 1):
                        return True
                    path.pop()
                    visited[nb] = False
            return False
        
        if backtrack2(2):
            return True
    
    return False

def _find_hamilton_cycle(adj):
    """Find an actual Hamilton cycle in the graph."""
    visited = [False] * N
    visited[0] = True
    path = [0]
    
    def backtrack(pos, prev):
        if pos == N:
            if 0 in adj[path[-1]]:
                return True
            return False
        
        for nb in adj[prev]:
            if not visited[nb]:
                visited[nb] = True
                path.append(nb)
                if backtrack(pos + 1, nb):
                    return True
                path.pop()
                visited[nb] = False
        return False
    
    for first_nb in sorted(adj[0]):
        visited = [False] * N
        visited[0] = True
        path = [0, first_nb]
        visited[first_nb] = True
        
        def backtrack2(pos):
            if pos == N:
                return 0 in adj[path[-1]]
            for nb in adj[path[-1]]:
                if not visited[nb]:
                    visited[nb] = True
                    path.append(nb)
                    if backtrack2(pos + 1):
                        return True
                    path.pop()
                    visited[nb] = False
            return False
        
        if backtrack2(2):
            return list(path)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Binary search on K: find minimum K such that prefix graph has Hamilton cycle
    lo, hi = 1, NUM_EDGES
    best_cycle = list(range(N))  # Fallback: identity cycle (score = max arrival of those edges)
    
    # First, compute the score of the identity cycle as fallback
    identity_score = 0
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        identity_score = max(identity_score, _edge_arrival.get((u, v), NUM_EDGES))
    
    # Binary search
    best_K = hi
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        adj = _get_adjacency(mid)
        
        if _has_hamilton_cycle(adj):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Find the actual cycle at best_K
    if time.time() < deadline - 2:
        adj = _get_adjacency(best_K)
        cycle = _find_hamilton_cycle(adj)
        if cycle is not None:
            return cycle
    
    # Fallback: try to improve identity cycle with local search
    # Or just return identity
    return best_cycle
# EVOLVE-BLOCK-END
