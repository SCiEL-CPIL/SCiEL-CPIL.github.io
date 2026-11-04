import random
import time
import numpy as np
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

def _girth(adj, g):
    """Check if graph has girth >= g using non-backtracking walks."""
    n = len(adj)
    if n == 0:
        return g
    
    from collections import deque
    
    # Build adjacency as sets for fast lookup
    adj_set = [set(adj[v]) for v in range(n)]
    
    # For each directed edge (u,v), do BFS in the state space of directed edges.
    # State: (prev, curr), distance = number of edges in the walk so far.
    # Start: (u, v) with distance 1.
    # Goal: reach (w, u) where w != v, with distance < g.
    # If such a walk exists, there's a cycle of length < g.
    
    for u in range(n):
        for v in adj[u]:
            visited = set()
            queue = deque()
            queue.append((u, v, 1))  # (prev, curr, dist)
            visited.add((u, v))
            
            while queue:
                prev, curr, d = queue.popleft()
                for next_v in adj_set[curr]:
                    if next_v == prev:
                        continue
                    # Check if we've reached a goal state: (w, u) where w != v
                    # curr is the current vertex, prev is the previous vertex
                    # We want curr == u and prev != v
                    if curr == u and prev != v:
                        if d < g:
                            return d
                    new_state = (curr, next_v)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((curr, next_v, d + 1))
    
    return g

def construct(k, g):
    deadline = time.time() + 110.0
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    # For k=5, g=9, we need n >= 428 (Moore bound lower bound)
    # The smallest known is 1116, but we'll try to find something smaller.
    # Let's start from a reasonable n and try random regular graphs with 2-switches.
    
    # Moore bound: n >= 2 + k * (k-1) * sum_{i=0}^{floor((g-2)/2)-1} (k-1)^i
    # For k=5, g=9: n >= 2 + 5*4*(1 + 4 + 16 + 64) = 2 + 20*85 = 1702? 
    # Wait, let me recalculate.
    # For odd girth g=2r+1, the Moore bound is n >= 2 + k * sum_{i=0}^{r-1} (k-1)^i
    # For g=9, r=4: n >= 2 + 5*(1 + 4 + 16 + 64) = 2 + 5*85 = 427
    # So n >= 428 (since n must be even for 5-regular).
    
    # Let's try n values starting from 428 upwards.
    # But random 5-regular graphs on 428 vertices likely have small girth.
    # We need to use 2-switches to increase the girth.
    
    # Strategy: start with a random 5-regular graph, then apply 2-switches
    # that increase the girth (i.e., break short cycles).
    
    n = 428
    max_n = 600  # Try up to 600 vertices
    
    while n <= max_n and time.time() < deadline:
        adj = None
        # Try multiple random initial graphs
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        
        if adj is not None:
            # Check initial girth
            girth = _girth(adj, g)
            if girth >= g:
                best = adj
                best_n = n
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # Apply 2-switches to try to increase girth
            # We want to break short cycles.
            # A 2-switch that breaks a short cycle is good.
            # Let's do random 2-switches and check if girth increases.
            
            best_girth = girth
            for _ in range(5000):
                if time.time() > deadline - 10:
                    break
                if _swap_regular(adj, n, k, rng):
                    new_girth = _girth(adj, g)
                    if new_girth >= g:
                        best = adj
                        best_n = n
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
                    elif new_girth > best_girth:
                        best_girth = new_girth
                    else:
                        # Revert? We don't have easy revert, so just continue
                        pass
        
        n += 2

    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: LUW construction
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
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
