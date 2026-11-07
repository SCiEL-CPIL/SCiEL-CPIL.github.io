"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_dist(adj, n, start):
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

def _check_valid(added_edges, n):
    """Check if added_edges make the graph triangle-free, connected, diameter <= 4."""
    all_edges = P['edges'] + added_edges
    adj = _build_adj(all_edges, n)
    
    # Check triangle-free
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
    
    # Check connected and diameter
    for start in range(n):
        dist = _bfs_dist(adj, n, start)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def _get_candidates(n, adj):
    """Get candidate edges that are triangle-free when added."""
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            # Check if adding (u,v) creates a triangle
            common = adj[u] & adj[v]
            if not common:
                candidates.append((u, v))
    return candidates

def construct():
    n = P['n']
    base_edges = P['edges']
    base_adj = _build_adj(base_edges, n)
    
    # Precompute candidate edges (triangle-free with base)
    candidates = _get_candidates(n, base_adj)
    
    # Greedy: repeatedly add edge that maximally reduces diameter/violations
    curr_adj = [set(s) for s in base_adj]
    added = []
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    while time.time() < deadline:
        # Check current diameter
        diam = 0
        connected = True
        for s in range(n):
            dist = _bfs_dist(curr_adj, n, s)
            if -1 in dist:
                connected = False
                break
            diam = max(diam, max(dist))
        
        if connected and diam <= 4:
            break
        
        if not connected:
            break
        
        # Find best edge to add
        best_edge = None
        best_diam = diam + 1
        
        for u, v in candidates:
            if v in curr_adj[u]:
                continue
            # Check triangle-free with current graph
            common = curr_adj[u] & curr_adj[v]
            if common:
                continue
            
            # Try adding
            curr_adj[u].add(v)
            curr_adj[v].add(u)
            
            new_diam = 0
            ok = True
            for s in range(n):
                dist = _bfs_dist(curr_adj, n, s)
                if -1 in dist:
                    ok = False
                    break
                new_diam = max(new_diam, max(dist))
            
            curr_adj[u].discard(v)
            curr_adj[v].discard(u)
            
            if ok and new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        curr_adj[u].add(v)
        curr_adj[v].add(u)
        added.append([int(u), int(v)])
    
    # Pruning: try to remove edges
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(added) - 1, -1, -1):
            u, v = added[i]
            curr_adj[u].discard(v)
            curr_adj[v].discard(u)
            
            if _check_valid(added[:i] + added[i+1:], n):
                added.pop(i)
                changed = True
            else:
                curr_adj[u].add(v)
                curr_adj[v].add(u)
    
    # Final verification
    if not _check_valid(added, n):
        # Fallback: return a known valid solution
        # Try a simple structure
        fallback = []
        for i in range(0, 24, 3):
            if i + 6 < 24:
                fallback.append([i, i + 6])
        if _check_valid(fallback, n):
            return fallback
        # Last resort: empty (invalid, but better than crash)
        return []
    
    return added
# EVOLVE-BLOCK-END
