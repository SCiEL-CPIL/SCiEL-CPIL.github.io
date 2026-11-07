"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 75.0  # safety margin before 90s kill

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution(n, base_edges, added_edges):
    edge_set = set()
    for u, v in base_edges:
        edge_set.add((min(u, v), max(u, v)))
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free check
    for i in range(n):
        for j in adj[i]:
            for k in adj[i]:
                if j < k and k in adj[j]:
                    return False
    
    # Connected + diameter
    for start in range(n):
        dist = _bfs_distances(adj, n, start)
        if -1 in dist or max(dist) > 4:
            return False
    
    return True

def _get_diameter(adj, n):
    """Compute diameter, return -1 if disconnected."""
    max_d = 0
    for start in range(n):
        dist = _bfs_distances(adj, n, start)
        if -1 in dist:
            return -1
        d = max(dist)
        if d > max_d:
            max_d = d
    return max_d

def _is_triangle_free_add(adj, n, u, v):
    """Check if adding edge (u,v) keeps graph triangle-free."""
    for k in adj[u]:
        if k != v and k in adj[v]:
            return False
    return True

def _build_adj(n, base_edges, added_edges):
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _try_pattern(n, base, base_set, strides):
    """Try adding edges (i, i+s) for each stride s, check validity."""
    added = []
    edge_set = set(base_set)
    for s in strides:
        for i in range(n - s):
            j = i + s
            if (i, j) not in edge_set:
                added.append([i, j])
                edge_set.add((i, j))
    if _check_solution(n, base, added):
        return added
    return None

def construct():
    n = P['n']
    base = P['edges']
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
    
    # Try various stride patterns
    stride_patterns = [
        [11],
        [11, 5],
        [11, 7],
        [10, 14],
        [9, 15],
        [8, 16],
        [7, 17],
        [6, 18],
        [5, 19],
        [4, 20],
        [11, 4],
        [11, 6],
        [11, 8],
        [12, 10],
        [13, 9],
        [14, 8],
        [15, 7],
    ]
    
    for strides in stride_patterns:
        if not _time_ok():
            break
        result = _try_pattern(n, base, base_set, strides)
        if result is not None:
            return result
    
    # Greedy with efficiency: prioritize edges that reduce the worst-case distance
    adj = [set() for _ in range(n)]
    for u, v in base:
        adj[u].add(v)
        adj[v].add(u)
    
    added = []
    edge_set = set(base_set)
    
    # Precompute all possible edges
    possible = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                possible.append((i, j))
    
    while _time_ok():
        diam = _get_diameter(adj, n)
        if diam <= 4:
            break
        
        # Find all pairs with distance > 4
        far_pairs = []
        for u in range(n):
            dist = _bfs_distances(adj, n, u)
            for v in range(u+1, n):
                if dist[v] > 4:
                    far_pairs.append((dist[v], u, v))
        
        if not far_pairs:
            break
        
        # Score each candidate edge by how many far pairs it might help
        best_edge = None
        best_score = -1
        
        for i, j in possible:
            if (i, j) in edge_set:
                continue
            if not _is_triangle_free_add(adj, n, i, j):
                continue
            
            # Score: count how many far pairs (u,v) could potentially be shortened
            # by adding (i,j). A rough heuristic: if i is on a shortest path between
            # some far pair, or j is.
            # Simpler: just check if adding this edge reduces the diameter
            adj[i].add(j)
            adj[j].add(i)
            new_diam = _get_diameter(adj, n)
            adj[i].remove(j)
            adj[j].remove(i)
            
            if new_diam == -1:
                continue
            
            # Prefer edges that reduce diameter most; tie-break by reducing far pairs
            score = -new_diam * 1000
            if score > best_score:
                best_score = score
                best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
    
    if _check_solution(n, base, added):
        return added
    
    # Fallback: return empty (invalid, but better than timeout)
    return []
# EVOLVE-BLOCK-END
