import time
import numpy as np
from array import array
from collections import deque

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
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    u = _randrange(rng, n)
    v = adj[u][_randrange(rng, k)]
    x = _randrange(rng, n)
    y = adj[x][_randrange(rng, k)]
    if (u == x and v == y) or (u == y and v == x):
        return False
    if u == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        return False
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    adj[u].append(x)
    adj[x].append(u)
    adj[v].append(y)
    adj[y].append(v)
    return True

def _girth_check_fast(adj, g):
    """
    Check if graph has girth >= g using non-backtracking walks.
    Returns True if girth >= g. Uses NumPy for speed.
    """
    n = len(adj)
    if n == 0:
        return True
    
    # Build adjacency matrix as 2D numpy array (padded to max degree)
    max_deg = max(len(a) for a in adj) if adj else 0
    if max_deg == 0:
        return True
    
    # Create padded adjacency array
    adj_arr = np.full((n, max_deg), -1, dtype=np.int32)
    for i in range(n):
        for j, nb in enumerate(adj[i]):
            adj_arr[i, j] = nb
    
    for start in range(n):
        # visited[prev, curr] = True if NB walk from start reached (prev,curr)
        visited = np.zeros((n, n), dtype=bool)
        
        # Initialize: start -> neighbors
        active_prev = np.array([start] * len(adj[start]), dtype=np.int32)
        active_curr = np.array(adj[start], dtype=np.int32)
        
        if len(active_curr) == 0:
            continue
        
        visited[active_prev, active_curr] = True
        
        depth = 1
        # We want to detect cycles of length < g
        # Check up to depth g-2 (cycles of length up to g-1)
        while len(active_curr) > 0 and depth < g - 1:
            # For each (prev, curr), expand to all neighbors of curr except prev
            next_prev_list = []
            next_curr_list = []
            
            for idx in range(len(active_curr)):
                prev = active_prev[idx]
                curr = active_curr[idx]
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start:
                        # Cycle of length depth + 1
                        if depth + 1 < g:
                            return False
                        continue
                    if not visited[curr, nxt]:
                        visited[curr, nxt] = True
                        next_prev_list.append(curr)
                        next_curr_list.append(nxt)
            
            if not next_prev_list:
                break
            active_prev = np.array(next_prev_list, dtype=np.int32)
            active_curr = np.array(next_curr_list, dtype=np.int32)
            depth += 1
    
    return True

def _luw_fallback(k, g):
    """LUW construction as fallback."""
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

def construct(k, g):
    deadline = time.time() + 115.0
    rng = 0x9E3779B97F4A7C15
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    
    n_start = max(2 * k, moore)
    if n_start % 2 != 0:
        n_start += 1
    
    # First, compute LUW as a safety net
    luw_n, luw_edges = _luw_fallback(k, g)
    
    # If LUW is already small, return it
    # Search for smaller graphs
    max_n = min(1200, luw_n - 2)  # Try to beat LUW
    
    if max_n < n_start:
        return luw_n, luw_edges
    
    n_values = list(range(n_start, max_n + 1, 2))
    n_values.sort()
    
    for n in n_values:
        if time.time() > deadline - 15:
            break
        
        # Try random graphs with 2-switch repair
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
                
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            
            # Quick girth check
            if _girth_check_fast(adj, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # Try 2-switches to fix short cycles
            for _ in range(200):
                if time.time() > deadline - 8:
                    break
                if _swap_regular(adj, n, k, rng):
                    if _girth_check_fast(adj, g):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
    
    # Fallback to LUW
    return luw_n, luw_edges
