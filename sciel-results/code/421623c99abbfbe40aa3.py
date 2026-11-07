"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_dist(adj, start):
    dist = [-1] * P['n']
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _diameter(adj):
    max_d = 0
    for i in range(P['n']):
        d = _bfs_dist(adj, i)
        if -1 in d:
            return 999
        max_d = max(max_d, max(d))
    return max_d

def _is_triangle_free(edge_set):
    adj = {i: set() for i in range(P['n'])}
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(P['n']):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u]:
                if w > v and w in adj[v]:
                    return False
    return True

def _build_adj(edge_set):
    adj = {i: set() for i in range(P['n'])}
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # Candidate edges: all non-base, non-self edges
    all_candidates = []
    for u in range(P['n']):
        for v in range(u + 1, P['n']):
            if (u, v) not in base_edges:
                all_candidates.append((u, v))
    
    # Greedy: start with path, add edges to reduce diameter
    current_edges = set(base_edges)
    adj = _build_adj(current_edges)
    current_diam = _diameter(adj)
    
    # Precompute which candidate edges create triangles
    # An edge (u,v) creates a triangle if there exists w adjacent to both u and v
    def creates_triangle(u, v, adj):
        for w in adj[u]:
            if w != v and w in adj[v]:
                return True
        return False
    
    while current_diam > 4 and time.time() < deadline:
        best_edge = None
        best_diam = current_diam
        # Sort candidates by potential benefit: prefer edges that connect distant vertices
        # Use a heuristic: edges that reduce max distance the most
        for u, v in all_candidates:
            if (u, v) in current_edges:
                continue
            if creates_triangle(u, v, adj):
                continue
            # Check if adding this edge helps
            new_adj = {i: set(s) for i, s in adj.items()}
            new_adj[u].add(v)
            new_adj[v].add(u)
            d = _diameter(new_adj)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
            if d == 4:
                break
            if time.time() > deadline:
                break
        if best_edge is None:
            break
        u, v = best_edge
        current_edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        current_diam = best_diam
    
    if current_diam > 4:
        # Fallback: use the bipartite complete graph (always valid)
        added = []
        for u in range(P['n']):
            for v in range(u + 1, P['n']):
                if (u, v) not in base_edges and (u - v) % 2 == 1:
                    added.append([u, v])
        return added
    
    # Removal phase: try to remove edges while maintaining constraints
    edge_list = list(current_edges)
    for i, (u, v) in enumerate(edge_list):
        if time.time() > deadline:
            break
        if (u, v) in base_edges:
            continue
        # Try removing
        current_edges.discard((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        # Check connected
        d = _diameter(adj)
        if d <= 4 and _is_triangle_free(current_edges):
            # Keep removed
            pass
        else:
            # Restore
            current_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    # Build result
    added = []
    for u, v in current_edges:
        if (u, v) not in base_edges:
            added.append([u, v])
    
    # Ensure types are correct
    result = []
    for e in added:
        result.append([int(e[0]), int(e[1])])
    
    return result

# EVOLVE-BLOCK-END
