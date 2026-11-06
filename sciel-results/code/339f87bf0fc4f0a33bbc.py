import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

# Build position map: edge -> arrival index (1-based)
_edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = _edge_key(u, v)
    _edge_pos[key] = i + 1

def _build_adjacency(K):
    """Build adjacency list for edges with arrival index <= K."""
    adj = [set() for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER):
        if i + 1 > K:
            break
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _warnsdorff_hamilton_cycle(adj, start=None, max_nodes=5000000):
    """Find a Hamilton cycle using Warnsdorff's heuristic.
    Returns the cycle as a list of vertices, or None if not found."""
    if start is None:
        # Try all starting vertices
        for s in range(N):
            cycle = _warnsdorff_dfs(adj, s, max_nodes)
            if cycle is not None:
                return cycle
        return None
    else:
        return _warnsdorff_dfs(adj, start, max_nodes)

def _warnsdorff_dfs(adj, start, max_nodes):
    """DFS with Warnsdorff's heuristic."""
    n = N
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute degree for initial ordering
    def get_order(node):
        neighbors = [v for v in adj[node] if not visited[v]]
        if not neighbors:
            return []
        # Warnsdorff: order by degree of unvisited neighbors
        neighbors.sort(key=lambda v: len([w for w in adj[v] if not visited[w]]))
        return neighbors
    
    def dfs(path):
        if len(path) == n:
            # Check if closing edge exists
            if path[-1] in adj[path[0]]:
                return True
            return False
        
        current = path[-1]
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return False
        
        # Warnsdorff's rule: visit neighbor with fewest unvisited neighbors first
        neighbors.sort(key=lambda v: len([w for w in adj[v] if not visited[w]]))
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            if dfs(path):
                return True
            path.pop()
            visited[v] = False
        return False
    
    if dfs(path):
        return path
    return None

def _cycle_score(cycle):
    """Compute the score (max arrival index) for a Hamilton cycle."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        key = _edge_key(u, v)
        idx = _edge_pos.get(key, float('inf'))
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _find_hamilton_cycle(K):
    """Try to find a Hamilton cycle in the graph with first K edges."""
    adj = _build_adjacency(K)
    
    # Quick check: every vertex must have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    # Try Warnsdorff from each starting vertex
    for s in range(N):
        cycle = _warnsdorff_dfs(adj, s, max_nodes=2000000)
        if cycle is not None:
            return cycle
    return None

def _two_opt_improve(cycle, edge_pos, max_iter=10000):
    """Try to improve a cycle's score using 2-opt style moves.
    This is a refinement step."""
    n = len(cycle)
    best_score = _cycle_score(cycle)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                # Try reversing segment i..j
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                # Check if valid (it's always a valid Hamilton cycle since it's a reversal)
                score = _cycle_score(new_cycle)
                if score < best_score:
                    cycle = new_cycle
                    best_score = score
                    improved = True
        if not improved:
            break
    return cycle, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s margin
    
    # Binary search on K
    lo = 24  # minimum edges needed for a Hamilton cycle
    hi = 276  # all edges
    best_cycle = None
    best_K = hi
    
    # First, try to find any cycle at full K (should always work)
    cycle = _find_hamilton_cycle(hi)
    if cycle is not None:
        best_cycle = cycle
        best_K = hi
    
    # Binary search for minimum K
    while lo < hi and (time.time() - start_time) < deadline - 5:
        mid = (lo + hi) // 2
        cycle = _find_hamilton_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Refinement: try to improve the best cycle found
    if best_cycle is not None and (time.time() - start_time) < deadline - 2:
        improved_cycle, improved_score = _two_opt_improve(best_cycle, _edge_pos, max_iter=50000)
        if improved_score < _cycle_score(best_cycle):
            best_cycle = improved_cycle
    
    # Final check
    if best_cycle is None:
        # Fallback: identity permutation
        return list(range(N))
    
    return best_cycle
# EVOLVE-BLOCK-END
