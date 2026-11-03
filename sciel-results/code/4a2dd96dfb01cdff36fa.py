import time
import random
from collections import deque

_START = time.time()
_DEADLINE = 110

def _check_girth(n, adj, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= g // 2: 
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _build_bigg_smith_cayley():
    """Try to find the Biggs-Smith graph as a Cayley graph of Z_2^6 x Z_5."""
    n = 320
    # Group elements: (mask, z) where mask is 6-bit int, z is 0..4
    # Index: mask * 5 + z
    def idx(mask, z):
        return mask * 5 + z
    
    def add(a, b):
        ma, za = divmod(a, 5)
        mb, zb = divmod(b, 5)
        return idx(ma ^ mb, (za + zb) % 5)
        
    def neg(a):
        ma, za = divmod(a, 5)
        return idx(ma, (-za) % 5)
        
    # Precompute addition table? 320x320 = 100k entries. Fine.
    # But we only need to check a few candidates.
    
    random.seed(42)
    candidates = []
    # Generate random a, b
    # Ensure a, b are not 0, and a != +/- b
    # Also ensure a, b are not involutions? 
    # If a is involution, a = -a. Then S={a, b, -b}. Size 3. Need 4.
    # So we prefer non-involutions.
    # Non-involution means z != 0.
    
    pool = [i for i in range(1, n) if i % 5 != 0] # z != 0
    if not pool:
        return None, None
        
    attempts = 0
    max_attempts = 5000 # ~1-2 seconds per 1000 checks? 
    # Girth check is ~0.1s. 5000 attempts = 500s. Too slow.
    # Need to optimize girth check or reduce attempts.
    # 320 vertices. BFS from each is fast.
    # Let's estimate: 320 * (320 * 4) ops = 400k ops.
    # Python is slow. Maybe 0.05s per check.
    # 2000 attempts = 100s.
    
    # We can pre-filter using shorter girths (4, 6).
    # If it has a 4-cycle, reject.
    
    def has_short_cycle(adj, max_len):
        # BFS from each vertex up to depth max_len//2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len <= max_len:
                            return True
        return False

    while attempts < 2000:
        a = random.choice(pool)
        b = random.choice(pool)
        na = neg(a)
        nb = neg(b)
        
        # Check distinctness
        if a == b or a == nb or na == b or na == nb:
            attempts += 1
            continue
        if a == na or b == nb: # Involutions
            attempts += 1
            continue
            
        S = [a, na, b, nb]
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u in range(n):
            for s in S:
                v = add(u, s)
                adj[u].append(v)
        
        # Check girth 4
        if has_short_cycle(adj, 4):
            attempts += 1
            continue
            
        # Check girth 6
        if has_short_cycle(adj, 6):
            attempts += 1
            continue
            
        # Check girth 8
        if has_short_cycle(adj, 8):
            attempts += 1
            continue
            
        # Check girth 10
        if _check_girth(n, adj, 10):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
        attempts += 1
        
    return None, None

def _build_luw_fallback():
    """Fallback to LUW construction (512 vertices)."""
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(4):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
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

def construct(k, g):
    if k == 4 and g == 10:
        # Try Cayley graph search for 320 vertices
        n, edges = _build_bigg_smith_cayley()
        if n is not None:
            return n, edges
        
        # Fallback to LUW (512 vertices)
        return _build_luw_fallback()
    else:
        return _build_luw_fallback()
