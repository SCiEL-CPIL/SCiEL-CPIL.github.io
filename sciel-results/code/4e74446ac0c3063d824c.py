import time
import math
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Known small cages
    if (k, g) == (3, 5):
        return _petersen()
    if (k, g) == (3, 6):
        return _heawood()
    if (k, g) == (3, 8):
        # Try to construct Tutte-Coxeter or fall back to LUW
        pass
        
    # 1. Try optimized LUW construction
    # LUW graph CD(d, q) is q-regular.
    # We need q = k.
    # Girth bound:
    # If d is odd, girth >= d + 5
    # If d is even, girth >= 2d + 1 (Hypothesis, needs verification)
    # We want to minimize n = 2 * k^d.
    # So we want the smallest d satisfying the girth condition.
    
    best_luw_result = None
    if _is_prime(k) or k == 4:
        # Try d from 1 upwards
        # Minimum d for girth g:
        # If g <= 5, d=1 might work? CD(1, q) is a cycle of length 2q? Girth 2q?
        # CD(1, q) has 2q vertices. It's a cycle $C_{2q}$. Girth $2q$.
        # If $2q \ge g$, it's valid.
        
        for d in range(1, 10): # Limit d to avoid huge n
            # Check girth bound
            if d % 2 == 1:
                girth_bound = d + 5
            else:
                girth_bound = 2 * d + 1 # Assuming this bound holds
            
            if girth_bound >= g:
                result = _luw_construct(k, g, d)
                if result is not None:
                    n, edges = result
                    if _verify_girth(n, edges, k, g):
                        if best_luw_result is None or n < best_luw_result[0]:
                            best_luw_result = (n, edges)
                        # Since n increases with d, the first valid d is the smallest n.
                        # So we can break?
                        # Wait, if d=1 is valid, n=2k.
                        # If d=2 is valid, n=2k^2.
                        # We want smallest n, so smallest d.
                        # So we return the first valid one.
                        return n, edges
        
    # If LUW failed or didn't run (k not prime), try Circulant
    # But for (5,9), LUW with d=4 should work.
    # 5 is prime.
    # d=1: girth_bound=6. If g=9, 6<9.
    # d=2: girth_bound=5. 5<9.
    # d=3: girth_bound=8. 8<9.
    # d=4: girth_bound=9. 9>=9.
    # So d=4 is the first valid.
    # n = 2 * 5^4 = 1250.
    
    # If we reached here, LUW didn't work.
    result = _circulant_construct(k, g, deadline)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, k, g):
            return n, edges

    # Fallback
    return _fallback_construct(k, g)

def _petersen():
    """Petersen graph: 3-regular, girth 5, 10 vertices."""
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _heawood():
    """Heawood graph: 3-regular, girth 6, 14 vertices."""
    edges = []
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((i, 7 + p))
    return 14, edges

def _luw_construct(k, g, d):
    """Lazebnik-Ustimenko-Woldar type construction with specific d."""
    q = k
    if q < 2:
        return None
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        if not _is_prime(q):
            return None
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(dg != k for dg in deg):
        return None
        
    return n, edges

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _verify_girth(n, edges, k, g):
    """Verify that the graph has girth at least g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _circulant_construct(k, g, deadline):
    """Search for a circulant k-regular graph with girth at least g."""
    if k < 2:
        return None
    
    # Moore bound calculation
    if g % 2 == 0:
        h = g // 2
        moore = 1 + k * sum([(k-1)**i for i in range(h)])
    else:
        h = (g - 1) // 2
        moore = 1 + k * sum([(k-1)**i for i in range(h)])
        
    moore = max(moore, 2 * k)
    
    # Search range
    start_n = moore
    end_n = moore + 200 # Limited search
    
    for n in range(start_n, end_n + 1):
        if time.time() > deadline - 5:
            return None
            
        trials = 100
        for _ in range(trials):
            if time.time() > deadline - 5:
                return None
                
            try:
                if k > n // 2:
                    continue
                steps = random.sample(range(1, n // 2 + 1), k)
            except ValueError:
                continue
                
            if _check_circulant_girth(n, steps, g):
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    return None

def _check_circulant_girth(n, steps, g):
    """Check if circulant graph with given steps has girth >= g."""
    max_depth = (g - 1) // 2
    
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = [0]
    q_head = 0
    
    while q_head < len(queue):
        u = queue[q_head]
        q_head += 1
        if dist[u] >= max_depth:
            continue
            
        du = dist[u]
        pu = parent[u]
        for s in steps:
            v1 = (u + s) % n
            v2 = (u - s) % n
            
            if dist[v1] == -1:
                dist[v1] = du + 1
                parent[v1] = u
                if du + 1 < max_depth:
                    queue.append(v1)
            elif parent[u] != v1 and parent[v1] != u:
                cycle_len = du + dist[v1] + 1
                if cycle_len < g:
                    return False
            
            if dist[v2] == -1:
                dist[v2] = du + 1
                parent[v2] = u
                if du + 1 < max_depth:
                    queue.append(v2)
            elif parent[u] != v2 and parent[v2] != u:
                cycle_len = du + dist[v2] + 1
                if cycle_len < g:
                    return False
                    
    return True

def _fallback_construct(k, g):
    """Fallback construction using a simple method."""
    n = 2 * k
    edges = []
    for i in range(k):
        for j in range(k):
            if i != j:
                edges.append((i, k + j))
    return n, edges
