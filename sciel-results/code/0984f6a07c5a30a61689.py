"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj, src):
    n = len(adj)
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _get_diameter(adj):
    n = len(adj)
    max_d = 0
    for i in range(n):
        dist = _bfs_distances(adj, i)
        for d in dist:
            if d > max_d:
                max_d = d
    return max_d

def _build_adj(n, edge_list):
    adj = [[] for _ in range(n)]
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_triangle_free_check(adj, u, v):
    nu = set(adj[u])
    nv = set(adj[v])
    return len(nu & nv) == 0

def _validate_solution(added_edges):
    n = P['n']
    all_edges = P['edges'] + added_edges
    # Check no duplicates
    edge_set = set()
    for u, v in all_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    adj = _build_adj(n, all_edges)
    # Check triangle-free
    for u in range(n):
        for i in range(len(adj[u])):
            for j in range(i+1, len(adj[u])):
                if adj[u][j] in adj[adj[u][i]]:
                    return False
    # Check connected
    dist = _bfs_distances(adj, 0)
    if any(d == -1 for d in dist):
        return False
    # Check diameter
    diam = _get_diameter(adj)
    if diam > 4:
        return False
    return True

def _construct_hub_based():
    n = P['n']
    # Hub at 11, connect to even vertices excluding 10, 12 (adjacent to 11)
    hub = 11
    candidates = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
    added = [[hub, c] for c in candidates]
    if _validate_solution(added):
        return added
    return added

def _try_remove_edges(added_edges):
    """Try to remove edges while maintaining validity."""
    n = P['n']
    current = list(added_edges)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            trial = current[:i] + current[i+1:]
            if _validate_solution(trial):
                current = trial
                improved = True
                break
    return current

def _construct():
    deadline = time.time() + 80
    n = P['n']
    
    # Start with hub-based solution
    added = _construct_hub_based()
    
    if time.time() > deadline:
        return [int(u) for u in []] if not added else [[int(u), int(v)] for u, v in added]
    
    # Try to reduce by removing edges
    added = _try_remove_edges(added)
    
    if time.time() > deadline:
        return [[int(u), int(v)] for u, v in added]
    
    # Try alternative hub positions
    for hub in [10, 11, 12, 13]:
        if time.time() > deadline:
            break
        candidates = [v for v in range(n) if v != hub and abs(v - hub) >= 2]
        # Sort by distance from hub, prefer far ones
        candidates.sort(key=lambda v: -abs(v - hub))
        # Greedily add edges from hub
        adj = _build_adj(n, P['edges'])
        trial_added = []
        for c in candidates:
            if not _is_triangle_free_check(adj, hub, c):
                continue
            adj[hub].append(c)
            adj[c].append(hub)
            trial_added.append([hub, c])
            if _get_diameter(adj) <= 4:
                break
        if len(trial_added) < len(added):
            if _validate_solution(trial_added):
                added = trial_added
                added = _try_remove_edges(added)
    
    return [[int(u), int(v)] for u, v in added]

def construct():
    result = _construct()
    # Ensure types are correct
    return [[int(u), int(v)] for u, v in result]
# EVOLVE-BLOCK-END
