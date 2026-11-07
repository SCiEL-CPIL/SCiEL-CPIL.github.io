"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edge_list):
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bipartition(n, adj):
    part = [-1] * n
    for s in range(n):
        if part[s] != -1:
            continue
        part[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if part[v] == -1:
                    part[v] = 1 - part[u]
                    q.append(v)
                elif part[v] == part[u]:
                    return None
    return part

def _is_bipartite(n, adj):
    return _bipartition(n, adj) is not None

def _diameter(n, adj):
    best = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if -1 in dist:
            return 999999
        best = max(best, max(dist))
    return best

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))

    adj = _build_adj(n, P['edges'])
    part = _bipartition(n, adj)

    # Partition A: even vertices (0,2,...,22), Partition B: odd (1,3,...,23)
    # We want to add edges only between A and B to keep triangle-free.
    # To minimize diameter, we want a "hub" in one partition connected to many in the other.
    # The star solution uses vertex 1 (in B) connected to all A.
    # That adds edges (1, a) for a in A \ {0, 2} (since (1,0) and (1,2) are already in path).
    # A = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}
    # Existing: (1,0), (1,2). So add (1,4), (1,6), (1,8), (1,10), (1,12), (1,14), (1,16), (1,18), (1,20), (1,22) = 10 edges.
    
    # Can we do better? Let's try a different hub or a two-hub structure.
    # Try hub at 0 (in A). Connect 0 to all B except 1 (already connected).
    # B = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
    # Existing: (0,1). Add (0,3), (0,5), ..., (0,23) = 11 edges. Worse.
    
    # Try hub at 23 (in B). Connect 23 to all A except 22 (already connected).
    # A = {0, 2, ..., 22}. Existing: (23,22). Add (23,0), (23,2), ..., (23,20) = 11 edges. Worse.
    
    # Try two hubs: 1 and 21.
    # 1 connects to some A, 21 connects to some A.
    # This is more complex. The single hub at 1 gives 10 edges.
    # Is 9 possible? 
    # With 9 edges added, total edges = 23 + 9 = 32.
    # The star gives diameter 4. 
    # Let's verify the star solution is valid and count edges.
    
    added_edges = []
    hub = 1
    for a in range(0, n, 2):
        if (hub, a) not in base_edges:
            added_edges.append([hub, a])
    
    # Verify the result
    all_edges = list(P['edges']) + added_edges
    adj2 = _build_adj(n, all_edges)
    
    if not _is_bipartite(n, adj2):
        # Fallback: should not happen
        return []
    
    if _diameter(n, adj2) > 4:
        # Fallback: should not happen
        return []
    
    return added_edges
# EVOLVE-BLOCK-END
