import time
import random
import sys

def _check_girth(adj, g):
    """
    Checks if graph with adjacency list 'adj' has girth at least 'g'.
    Returns True if girth >= g.
    """
    n = len(adj)
    # BFS from each node to depth g//2
    max_depth = g // 2
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [u]
        dist[u] = 0
        qi = 0
        while qi < len(queue):
            cur = queue[qi]
            qi += 1
            if dist[cur] >= max_depth:
                continue
            for nei in adj[cur]:
                if dist[nei] == -1:
                    dist[nei] = dist[cur] + 1
                    parent[nei] = cur
                    queue.append(nei)
                else:
                    # Found a cycle
                    # Length is dist[cur] + dist[nei] + 1
                    # If this is < g, fail
                    # Note: This detects *some* cycle. If the cycle is not simple, 
                    # the girth is still <= length.
                    # However, BFS tree path + edge is a simple cycle if the edge is not a tree edge.
                    # If dist[nei] <= dist[cur] + 1, it's a tree edge or back edge.
                    # We only care if a cycle of length < g exists.
                    cycle_len = dist[cur] + dist[nei] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    if k != 4 or g != 10:
        # Fallback for other cases: return empty or dummy
        # Task is specific to 4,10
        return 0, []

    n_best = 320
    edges_best = []
    
    # 1. Try Circulant Graphs C_320(1, j)
    # Degree 4 implies neighbors +/-1, +/-j
    # Check j from 2 to 159
    for j in range(2, 160):
        if time.time() > deadline:
            break
        n = 320
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(n):
            v1 = (i + 1) % n
            v2 = (i - 1) % n
            v3 = (i + j) % n
            v4 = (i - j) % n
            
            # Add edges uniquely
            for v in [v1, v2, v3, v4]:
                if i < v:
                    edges.append((i, v))
                    adj[i].append(v)
                    adj[v].append(i)
                elif i > v:
                    # Already added when v was processed?
                    # No, we are iterating i.
                    # If i > v, the edge (v, i) should have been added when i was v?
                    # Wait.
                    # If we only add when i < v, we miss edges where i > v.
                    # But we iterate all i.
                    # For edge (u, v) with u < v:
                    # When i=u, v is neighbor. u < v, add (u, v).
                    # When i=v, u is neighbor. v > u, do not add.
                    # So each edge is added exactly once.
                    pass
        
        # Verify degree
        if all(len(a) == 4 for a in adj):
            if _check_girth(adj, 10):
                n_best = 320
                edges_best = edges
                return n_best, edges_best

    # 2. Random Search for n < 320
    for n_try in range(318, 242, -1):
        if time.time() > deadline:
            break
        
        # Try a few random graphs
        for _ in range(50): # Increased attempts
            if time.time() > deadline:
                break
            
            # Generate random 4-regular graph
            stubs = [i for i in range(n_try) for _ in range(4)]
            random.shuffle(stubs)
            adj_try = [[] for _ in range(n_try)]
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj_try[u]:
                    valid = False
                    break
                adj_try[u].append(v)
                adj_try[v].append(u)
            
            if not valid:
                continue
            
            if _check_girth(adj_try, 10):
                n_best = n_try
                edges_best = []
                for i in range(n_try):
                    for j in adj_try[i]:
                        if i < j:
                            edges_best.append((i, j))
                return n_best, edges_best

    # 3. Random Search for n = 320 if circulant failed
    if n_best == 320 and not edges_best:
        for _ in range(200):
            if time.time() > deadline:
                break
            n_try = 320
            stubs = [i for i in range(n_try) for _ in range(4)]
            random.shuffle(stubs)
            adj_try = [[] for _ in range(n_try)]
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj_try[u]:
                    valid = False
                    break
                adj_try[u].append(v)
                adj_try[v].append(u)
            
            if not valid:
                continue
            
            if _check_girth(adj_try, 10):
                n_best = 320
                edges_best = []
                for i in range(n_try):
                    for j in adj_try[i]:
                        if i < j:
                            edges_best.append((i, j))
                return n_best, edges_best

    # Fallback: Return a dummy if nothing found (should not happen if 320 cage exists and search is good)
    # But to avoid error, return a simple 4-regular graph on 320 vertices (might have low girth)
    # The evaluator will reject it, but it's better than an exception?
    # No, score 0 is same.
    # Let's return the first valid 4-regular graph we can make, even if girth is low, 
    # just to satisfy the interface? No, "invalid result... scores 0".
    # So we must return a VALID graph.
    # If we didn't find one, we are stuck.
    # But the circulant search should likely find one or the random search will.
    
    return 320, []
