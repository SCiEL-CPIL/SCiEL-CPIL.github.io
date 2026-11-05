import time
import random
import sys
from collections import deque

_DEADLINE = 145.0
_START = 0.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _girth_check(adj, g):
    """Returns True if girth >= g. Uses BFS from each vertex."""
    n = len(adj)
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _random_4regular(n, seed):
    rng = random.Random(seed)
    for _ in range(50):
        half = list(range(n)) * 4
        rng.shuffle(half)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, 4*n, 2):
            a, b = half[i], half[i+1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if not ok: continue
        if any(len(s) != 4 for s in adj): continue
        return [list(s) for s in adj]
    return None

def _local_search(adj, g, time_limit):
    n = len(adj)
    start_time = time.time()
    rng = random.Random(int(start_time * 1000))
    
    # Convert to sets for O(1) lookup
    adj_sets = [set(lst) for lst in adj]
    
    while time.time() - start_time < time_limit:
        if _girth_check([list(s) for s in adj_sets], g):
            return [list(s) for s in adj_sets]
        
        # Find a short cycle to break
        # We can just pick a random edge and swap
        # Or find the actual short cycle.
        # For simplicity, random edge swap.
        u = rng.randrange(n)
        v = rng.choice(adj_sets[u])
        x = rng.randrange(n)
        if x == u or x == v: continue
        if x in adj_sets[u]: continue
        
        # Find y neighbor of x
        y = rng.choice(adj_sets[x])
        if y == u or y == v: continue
        if y in adj_sets[v]: continue
        if x in adj_sets[v]: continue # This would create a triangle? No, just check validity
        if u in adj_sets[y]: continue
        
        # Perform swap (u,v) and (x,y) -> (u,x) and (v,y)
        # Check if this creates a multi-edge or loop
        if x in adj_sets[u]: continue
        if y in adj_sets[v]: continue
        
        adj_sets[u].remove(v)
        adj_sets[v].remove(u)
        adj_sets[x].remove(y)
        adj_sets[y].remove(x)
        
        adj_sets[u].add(x)
        adj_sets[x].add(u)
        adj_sets[v].add(y)
        adj_sets[y].add(v)
        
    return None

def construct(k, g):
    global _START
    _START = time.time()
    
    if k != 4:
        return _luw(k, g)
        
    best_adj = None
    best_n = None
    
    # Try n from 242 to 320
    for n in range(242, 321, 2):
        if _remaining() < 5:
            break
            
        found = False
        start_n_time = time.time()
        
        # Budget per n: start small, increase if we have time
        # But we want to find the smallest n.
        # Let's give a fixed small budget per n to scan through.
        # If we find one, return.
        
        while time.time() - start_n_time < 0.5 and _remaining() > 2:
            adj = _random_4regular(n, random.randrange(1<<30))
            if adj is None: continue
            if _girth_check(adj, g):
                best_adj = adj
                best_n = n
                found = True
                break
            # Local search
            adj_ls = _local_search(adj, g, 0.2)
            if adj_ls is not None:
                best_adj = adj_ls
                best_n = n
                found = True
                break
        
        if found:
            edges = []
            for i in range(n):
                for j in best_adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    return _luw(k, g)

def _luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction (valid fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
    return len(order), edges
