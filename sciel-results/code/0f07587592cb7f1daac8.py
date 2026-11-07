import time
import sys
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

def _is_triangle_free(adj, u, v):
    nu = set(adj[u])
    nv = set(adj[v])
    common = nu & nv
    return len(common) == 0

def _construct_greedy():
    n = P['n']
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    added_edges = []
    deadline = time.time() + 55  # 55s safety margin
    
    while time.time() < deadline:
        diam = _get_diameter(adj)
        if diam <= 4:
            break
        
        # Find the pair with max distance
        max_d = -1
        best_pair = None
        for i in range(n):
            dist = _bfs_distances(adj, i)
            for j in range(i+1, n):
                if dist[j] > max_d:
                    max_d = dist[j]
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        best_edge = None
        best_new_diam = diam + 1
        
        # Candidate edges: all non-existing, triangle-free edges
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edge_set:
                    continue
                # Check triangle-free
                if not _is_triangle_free(adj, u, v):
                    continue
                
                # Try adding this edge
                adj[u].append(v)
                adj[v].append(u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                
                new_diam = _get_diameter(adj)
                
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.remove((u, v))
                edge_set.remove((v, u))
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
        added_edges.append([u, v])
        
        # If we have a good solution (<= 6 edges), we might stop early if time is tight
        # But let's try to minimize further if time allows
        
    return added_edges

def construct():
    return _construct_greedy()
