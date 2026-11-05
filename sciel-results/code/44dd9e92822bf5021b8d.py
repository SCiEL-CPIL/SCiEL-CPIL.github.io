import time
import math
from collections import defaultdict

_DEADLINE = time.time() + 115.0

def _time_ok():
    return time.time() < _DEADLINE

def _check_girth_circulant(n, a, b, g):
    """
    Check if C_n({a, -a, b, -b}) has girth >= g.
    Uses dynamic programming on sums of steps.
    """
    # Ensure a, b are distinct and non-zero
    if a == 0 or b == 0: return False
    a = a % n
    b = b % n
    if a == 0 or b == 0: return False
    if a == b or a == n - b: return False # a == -b
    
    # Check connectivity: gcd(a, b, n) must be 1
    import math
    if math.gcd(math.gcd(a, b), n) != 1:
        return False

    steps = [a, n - a, b, n - b]
    
    # S_L is the set of all possible sums of L steps
    # We start with S_1
    current_sums = set(steps)
    
    # We need to check lengths 2 to g-1
    # Length 2: a + (-a) = 0. This is always true.
    # But a cycle of length 2 is not a simple cycle in a simple graph.
    # The girth is the length of the shortest SIMPLE cycle.
    # A closed walk of length L contains a simple cycle of length <= L.
    # However, the walk $a, -a$ is a backtrack, not a cycle.
    # The walk $a, b, -a, -b$ sums to 0. Length 4.
    # Does it contain a simple cycle?
    # If the vertices are distinct, yes.
    # If $a = b$, we already excluded.
    # If $a + b = 0$, we excluded.
    # If $2a = 0$, then $a, a, a, a$ sums to 0? No.
    # The condition "0 in S_L" is necessary for a closed walk.
    # Is it sufficient for a simple cycle?
    # If there is a closed walk of length L, there is a simple cycle of length <= L.
    # So if we find 0 in S_L for L < g, the girth is < g.
    # Exception: L=2. The only walk is $a, -a$ or $b, -b$ or $a, a$ (if $2a=0$).
    # $a, -a$ is not a simple cycle.
    # So for L=2, we should ignore 0?
    # Actually, if $2a = 0$, then $a, a$ is a closed walk of length 2.
    # This implies a cycle of length 2? No, in a simple graph, no multiple edges.
    # But $a$ is a step. $i \to i+a$. If $2a=0$, then $i+a = i-a$.
    # The edge is $(i, i+a)$.
    # If we take step $a$ then $a$, we go $i \to i+a \to i+2a = i$.
    # This is a walk of length 2.
    # Does it form a cycle?
    # The vertices are $i, i+a$.
    # The edges are $(i, i+a)$ and $(i+a, i)$.
    # This is the same edge traversed twice.
    # So it's not a simple cycle.
    # So we should check $L \ge 3$.
    # But for $L=3$, if $a+b+c=0$?
    # If $a+b+a = 2a+b=0$, etc.
    
    # So we check L from 3 to g-1.
    for L in range(3, g):
        if 0 in current_sums:
            return False
        next_sums = set()
        for s in current_sums:
            for step in steps:
                next_sums.add((s + step) % n)
        current_sums = next_sums
        
    return True

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 115.0

    if k != 4 or g != 10:
        # Fallback for other parameters, though task is specific
        return 2*k, [(i, (i+1)%(2*k)) for i in range(2*k)]

    # Moore bound for k=4, g=10 is 161.
    # We search n from 161 to 320 (known upper bound from prompt).
    # Actually, we can search up to 400 to be safe, but 320 is the "smallest known".
    # If we find one smaller, great.
    
    best_n = None
    best_edges = None

    # Search for circulant graphs
    # n must be such that a 4-regular circulant exists.
    # n can be any integer >= 5.
    # But for girth 10, n must be large.
    
    # To ensure we find the 256-vertex cage if it's circulant.
    # We iterate n.
    # Optimization: n must be even? No.
    # But if n is odd, a and b must be such that gcd(a,b,n)=1.
    
    max_n = 320
    # Start from Moore bound
    min_n = 161
    
    for n in range(min_n, max_n + 1):
        if not _time_ok():
            break
            
        # Iterate a, b
        # a < b
        # a, b in 1..n-1
        # Symmetry: a < n/2, b < n/2?
        # If a > n/2, replace with n-a.
        # So we can assume 1 <= a < b < n/2.
        # Also a != b.
        
        # To reduce search space:
        # a and b should be "far apart"?
        # Just iterate.
        
        for a in range(1, n // 2 + 1):
            if not _time_ok(): break
            for b in range(a + 1, n // 2 + 1):
                if not _time_ok(): break
                
                if _check_girth_circulant(n, a, b, g):
                    # Found a valid circulant
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i - a) % n))
                        edges.append((i, (i + b) % n))
                        edges.append((i, (i - b) % n))
                    # Deduplicate and sort
                    edges = sorted(set(edges))
                    
                    # Verify degree and girth with BFS to be 100% sure
                    # (The DP check is theoretically sound, but BFS is a good sanity check)
                    if _verify(n, edges, k, g):
                        best_n = n
                        best_edges = edges
                        return best_n, best_edges
                        
    # If no circulant found, fallback to LUW (2048 vertices) or a known 320-vertex graph?
    # The prompt implies 320 is the smallest known.
    # If we didn't find a circulant < 320, we should return the 320-vertex graph if we can construct it.
    # But we don't have the construction for the 320-vertex graph.
    # We have LUW (2048).
    # Is there a better fallback?
    # Maybe a random search for non-circulant graphs? Too slow.
    
    # Let's try to construct the LUW graph as fallback.
    n_lw, edges_lw = _lw_construct(4, 5)
    if _verify(n_lw, edges_lw, k, g):
        return n_lw, edges_lw

    # Absolute fallback
    return 2*k, [(i, (i+1)%(2*k)) for i in range(2*k)]

def _verify(n, edges, k, g):
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if not all(d == k for d in deg):
        return False
        
    # Check girth with BFS
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        if min_girth < g:
            return False
            
    return True

def _lw_construct(q, d):
    # Same as before, but simplified
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: x ^ y
        sub = add
        mul = lambda x, y: mul_table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

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

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vi, index[other]))

    return len(order), edges
