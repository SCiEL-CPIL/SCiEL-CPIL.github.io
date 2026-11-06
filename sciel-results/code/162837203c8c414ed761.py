import time
import random

def _build_polarity_graph_q5():
    """Build polarity graph from PG(2,5). 31 vertices, C4-free."""
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    n = len(points)
    
    a = 2  # non-square mod 5
    
    def bilinear(p, r):
        x1, y1, z1 = p
        x2, y2, z2 = r
        return (x1*y2 + y1*x2 + z1*z2 + a*(x1*z2 + z1*x2)) % q
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if bilinear(points[i], points[j]) == 0:
                edges.append([i, j])
    
    return n, edges

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4. adj[i] is bitmask of neighbors of i."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if common & (common - 1):
                return True
    return False

def _max_independent_set_bitmask(n, adj):
    """Maximum independent set size via branch and bound."""
    best = [0]
    
    def search(remaining, size):
        if not remaining:
            if size > best[0]:
                best[0] = size
            return
        if size + remaining.bit_count() <= best[0]:
            return
        v = (remaining & -remaining).bit_length() - 1
        search(remaining & ~(1 << v) & ~adj[v], size + 1)
        search(remaining & ~(1 << v), size)
    
    search((1 << n) - 1, 0)
    return best[0]

def _construct_large_graph():
    start = time.time()
    deadline = 100
    
    n, edges = _build_polarity_graph_q5()
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return None
    
    alpha = _max_independent_set_bitmask(n, adj)
    if alpha >= 11:
        return None
    
    best_n = n
    best_edges = list(edges)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        S = []
        for v in order:
            ok = True
            for u in S:
                if (adj[u] & adj[v]).bit_count() >= 2:
                    ok = False
                    break
            if ok:
                S.append(v)
        
        if len(S) < 2:
            continue
        
        new_n = n + 1
        new_adj = adj[:] + [0]
        for v in S:
            new_adj[v] |= (1 << new_n)
            new_adj[new_n] |= (1 << v)
        
        if _has_c4_bitmask(new_n, new_adj):
            continue
        
        new_alpha = _max_independent_set_bitmask(new_n, new_adj)
        if new_alpha < 11:
            n = new_n
            adj = new_adj
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

def construct():
    start_time = time.time()
    deadline = 110
    
    try:
        result = _construct_large_graph()
        if result and time.time() < deadline:
            return result
    except Exception:
        pass
    
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
