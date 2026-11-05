"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
NUM_EDGES = len(EDGE_ORDER)

# Build edge index lookup: (u,v) -> 1-based arrival index
_EDGE_INDEX = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    if u > v:
        u, v = v, u
    _EDGE_INDEX[(u, v)] = i + 1

def _edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return _EDGE_INDEX.get((u, v), NUM_EDGES + 1)

def _cycle_score(perm):
    """Score = max arrival index of edges in the cycle (with closing edge)."""
    max_idx = 0
    n = len(perm)
    for i in range(n):
        u = perm[i]
        v = perm[(i + 1) % n]
        idx = _edge_arrival(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _build_adjacency(K):
    """Build adjacency list using only edges with arrival <= K."""
    adj = [set() for _ in range(N)]
    for i in range(min(K, NUM_EDGES)):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_hamilton_cycle(adj, deadline):
    """
    Check if graph with adjacency 'adj' has a Hamilton cycle using backtracking.
    Uses pruning: degree >= 2, connectivity, articulation point checks.
    Returns True if found, False otherwise.
    """
    n = N
    
    # Quick degree check: every vertex must have degree >= 2
    for v in range(n):
        if len(adj[v]) < 2:
            return False
    
    # Connectivity check via BFS
    visited = [False] * n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    if count != n:
        return False
    
    # Backtracking: fix vertex 0 as start, try all orderings
    # State: current vertex, set of used vertices, path so far
    # We build a path of length n and check if last connects back to first.
    
    used = [False] * n
    path = [0]
    used[0] = True
    
    sys.setrecursionlimit(10000)
    
    def _backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check closing edge
            last = path[-1]
            first = path[0]
            if first in adj[last]:
                return True
            return False
        
        cur = path[-1]
        remaining = n - len(path)
        
        # Get candidates: neighbors of cur that are unused
        # If we're at the second-to-last step, the next vertex must connect back to start
        candidates = []
        for nb in adj[cur]:
            if not used[nb]:
                candidates.append(nb)
        
        if not candidates:
            return False
        
        # Pruning: if remaining == 2, the next vertex must be able to connect to start
        if remaining == 2:
            # path will be [..., cur, nb, last] where last must connect to start
            # Actually: we pick nb, then need one more vertex that connects to both nb and start
            filtered = []
            for nb in candidates:
                # After picking nb, we need one more vertex w such that w in adj[nb] and w in adj[start] and not used
                start = path[0]
                for w in adj[nb]:
                    if not used[w] and w != nb and w in adj[start]:
                        filtered.append(nb)
                        break
            if not filtered:
                return False
            candidates = filtered
        
        # Pruning: check that unused vertices still have degree >= 2 in the induced subgraph
        # (considering that used vertices' edges are "consumed" for internal path)
        # This is expensive, so do it only at certain depths
        if len(path) % 4 == 0:
            for v in range(n):
                if not used[v]:
                    deg = 0
                    for nb in adj[v]:
                        if not used[nb] or nb == path[-1]:
                            # nb is available or is the current vertex (which will be "used" after this step)
                            if not used[nb]:
                                deg += 1
                    # Also count edges to the current vertex (which will become used)
                    # Actually simpler: just check that in the remaining subgraph + current, deg >= 2
                    # Let's just do a basic check: each unused vertex must have at least 2 neighbors among unused or {cur}
                    if deg < 1:  # at minimum needs to connect to cur or another unused
                        return False
        
        # Order candidates by degree (fewest options first - MRV heuristic)
        candidates.sort(key=lambda x: len(adj[x]))
        
        for nb in candidates:
            used[nb] = True
            path.append(nb)
            if _backtrack():
                return True
            path.pop()
            used[nb] = False
            if time.time() > deadline:
                return False
        
        return False
    
    return _backtrack()

def _find_best_cycle(deadline):
    """
    Binary search on K to find the minimum threshold such that a Hamilton cycle exists.
    Then extract the actual cycle.
    """
    # First, find minimum K using binary search
    lo = N  # minimum possible: need at least N edges
    hi = NUM_EDGES  # worst case: all edges
    
    best_K = None
    # Binary search for minimum K where Hamilton cycle exists
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        adj = _build_adjacency(mid)
        if _has_hamilton_cycle(adj, deadline - 5):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_K is None:
        # Fallback: use full graph
        best_K = NUM_EDGES
    
    # Now extract the actual cycle from the graph with threshold best_K
    adj = _build_adjacency(best_K)
    cycle = _extract_cycle(adj, deadline)
    
    if cycle is None:
        # Fallback to identity
        return list(range(N))
    
    return cycle

def _extract_cycle(adj, deadline):
    """Extract a Hamilton cycle from the adjacency using backtracking."""
    n = N
    used = [False] * n
    path = [0]
    used[0] = True
    sys.setrecursionlimit(10000)
    
    def _backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            if path[0] in adj[path[-1]]:
                return list(path)
            return None
        
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if not used[nb]]
        
        if not candidates:
            return None
        
        # If one more to go, must connect back to start
        if len(path) == n - 1:
            candidates = [nb for nb in candidates if path[0] in adj[nb]]
            if not candidates:
                return None
        
        candidates.sort(key=lambda x: len(adj[x]))
        
        for nb in candidates:
            used[nb] = True
            path.append(nb)
            result = _backtrack()
            if result is not None:
                return result
            path.pop()
            used[nb] = False
            if time.time() > deadline:
                return None
        
        return None
    
    return _backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin within 90s limit
    
    # Try to find a good cycle
    result = _find_best_cycle(deadline)
    
    # Validate: must be a permutation of 0..23
    if result is None or len(result) != N or set(result) != set(range(N)):
        result = list(range(N))
    
    return result
# EVOLVE-BLOCK-END
