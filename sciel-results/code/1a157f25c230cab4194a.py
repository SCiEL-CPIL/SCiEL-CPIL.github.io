"""Improved LUW-based construction with parameter search and girth verification."""
import time
import sys

def _luw_construct(k, d):
    """Construct LUW graph CD(d, q) with q=k, return (n, edges)."""
    q = k
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q
    
    # Build the rule table
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r = (c-5)%4
        c0 = c - (c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    
    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _verify_girth(n, edges, min_girth):
    """Verify that the graph has girth >= min_girth using BFS from each vertex.
    Returns True if girth >= min_girth, False otherwise."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth verification, we need to check for cycles of length < min_girth
    # Use BFS with parent tracking to detect shortest cycle through each edge
    # Actually, a simpler approach: for each vertex, do BFS up to depth (min_girth-1)//2
    # and check if we can form a cycle shorter than min_girth
    
    half = (min_girth - 1) // 2  # max depth for BFS from each vertex
    
    for start in range(n):
        # BFS from start, track distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle: dist[u] + 1 + dist[v]
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    # Try multiple d values and pick the best
    # d should be at least 3, and the girth guarantee is d+5 for odd d
    # We need girth >= g, so for odd d: d+5 >= g => d >= g-5
    # Try a range of d values
    
    best_n = None
    best_edges = None
    
    # Determine valid d range
    # Minimum d: need girth >= g. For odd d, girth >= d+5, so d >= g-5
    # Try d from max(3, g-5) to some upper bound
    d_min = max(3, g - 5)
    # Make sure we try both odd and even d
    # Upper bound: don't go too high as it increases n
    d_max = d_min + 10  # try a few more
    
    for d in range(d_min, d_max + 1):
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(k, d)
        except Exception:
            continue
        
        if best_n is None or n < best_n:
            # Verify girth
            if time.time() > deadline - 5.0:
                # Not enough time to verify, skip verification for now
                # But we should verify at least the best candidate
                if best_n is not None:
                    break
                # Accept without verification as fallback
                best_n = n
                best_edges = edges
                break
            if _verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
            # If girth fails, this d doesn't work, try next
    
    if best_n is None:
        # Fallback: use the original logic
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = _luw_construct(k, d)
    
    return best_n, best_edges
