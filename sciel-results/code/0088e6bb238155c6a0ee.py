import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Calculate Moore Bound
    if g % 2 == 0:
        r = g // 2
        if k == 2:
            moore = g
        else:
            moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        if k == 2:
            moore = g
        else:
            moore = 1 + k * sum((k-1)**i for i in range(r))
            
    # Ensure n is even if k is odd
    if k % 2 != 0 and moore % 2 != 0:
        moore += 1
        
    n = moore
    
    while time.time() < deadline:
        max_tries = 10000
        found = False
        
        for _ in range(max_tries):
            if time.time() > deadline:
                break
            edges = generate_random_regular_graph(k, n)
            if edges is None:
                continue
            if has_girth_at_least(g, n, edges):
                return n, edges
        
        n += 2 if (k % 2 != 0) else 1

    # Fallback
    n_large = max(moore * 2, 100)
    while time.time() < deadline:
        edges = generate_random_regular_graph(k, n_large)
        if edges and has_girth_at_least(g, n_large, edges):
            return n_large, edges
        n_large += 10
        
    edges = generate_random_regular_graph(k, max(moore, 10))
    if edges is None:
        return 0, []
    return n, edges

def generate_random_regular_graph(k, n):
    if n * k % 2 != 0:
        return None
    half_edges = [i for i in range(n) for _ in range(k)]
    random.shuffle(half_edges)
    edges = []
    seen = set()
    for i in range(0, n * k, 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return None
        seen.add((u, v))
        edges.append((u, v))
    return edges

def has_girth_at_least(g, n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [start]
        dist[start] = 0
        
        for x in queue:
            if dist[x] >= g // 2:
                continue
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    parent[y] = x
                    queue.append(y)
                else:
                    if y != parent[x]:
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
    return True
