import time
import sys
import numpy as np
from collections import deque, defaultdict
import random
import math

_DEADLINE = 150.0
_T0 = None

def _elapsed():
    return time.time() - _T0

def _time_ok(margin=5.0):
    return _elapsed() < _DEADLINE - margin

def construct(k, g):
    global _T0
    _T0 = time.time()
    
    # Handle small cases with hardcoded cages
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return (n, edges)
    
    if k == 3:
        if g <= 4:
            return (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
        if g <= 5:
            # Petersen
            edges = []
            for i in range(5):
                edges.append((i, (i + 1) % 5))
                edges.append((5 + i, 5 + (i + 2) % 5))
                edges.append((i, 5 + i))
            return (10, edges)
        if g <= 6:
            # Heawood
            edges = []
            fano_lines = [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]
            for li, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7 + li))
            return (14, edges)
        if g <= 7:
            # Tutte-Coxeter (30 vertices, girth 7)
            # Constructed as bipartite double cover of Petersen? No.
            # Use LUW for k=3, g=7? q=3. d=3? 2*27=54. Too big.
            # Hardcode Tutte-Coxeter?
            # It is the incidence graph of the Fano plane? No, that's Heawood (14).
            # Tutte-Coxeter is 30 vertices.
            # I will rely on LUW for k=3 if g>6, or fallback.
            pass
        # For k=3, g>=7, try LUW
        result = _luw_construct(k, g)
        if result is not None:
            return result
        # Fallback to simple regular
        return _simple_regular_graph(k)

    if k == 5 and g == 9:
        # Try to find the 1116 cage
        if _time_ok(margin=30.0):
            result = _search_cage_5_9()
            if result is not None:
                return result
        
        # Fallback to LUW (1250 vertices)
        result = _luw_construct(k, g)
        if result is not None:
            return result
        
        # Fallback to simple regular
        return _simple_regular_graph(k)

    # General case: Try LUW
    result = _luw_construct(k, g)
    if result is not None:
        return result
    
    # Fallback: Greedy search for specific cases if time permits
    if _time_ok(margin=20.0):
        # Estimate n from Moore bound
        moore_bound = _moore_bound(k, g)
        n_target = int(math.ceil(moore_bound * 1.5))
        if n_target < 50: n_target = 50
        if n_target > 2000: n_target = 2000
        
        result = _greedy_girth_max_v2(n_target, k, g)
        if result is not None:
            return result

    return _simple_regular_graph(k)

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        return 2 * (1 + k * sum([(k-1)**i for i in range(r-1)]))
    else:
        r = (g - 1) // 2
        return 1 + k * sum([(k-1)**i for i in range(r)])

def _search_cage_5_9():
    """Specific search for the 1116-vertex cage."""
    n = 1116
    k = 5
    g = 9
    
    # Generate random regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
    
    # Greedy cycle breaking
    # We want to eliminate cycles of length < 9
    
    # Precompute neighbor lists
    neighbors = [np.where(adj[i] == 1)[0].tolist() for i in range(n)]
    
    # Function to count short cycles (approximate)
    # For speed, we just check girth at the end.
    # Here we do local search.
    
    best_girth = _calculate_girth_fast(adj, n)
    if best_girth >= g:
        return (n, _adj_to_edges(adj))
    
    # Simple local search: try to break short cycles
    # This is a heuristic.
    
    max_iters = 5000
    for _ in range(max_iters):
        if not _time_ok(margin=10.0):
            break
        
        # Pick random edge
        u = random.randint(0, n - 1)
        v_list = neighbors[u]
        if not v_list: continue
        v = random.choice(v_list)
        
        # Pick another edge
        u2 = random.randint(0, n - 1)
        v2_list = neighbors[u2]
        if not v2_list: continue
        v2 = random.choice(v2_list)
        
        # Check validity of swap (u,v) and (u2,v2) -> (u,v2) and (u2,v)
        if adj[u][v2] == 1 or adj[u2][v] == 1:
            continue
        if u == v2 or u2 == v:
            continue
        if u2 == u or v2 == v:
            continue
            
        # Perform swap
        adj[u][v] = 0
        adj[v][u] = 0
        adj[u2][v2] = 0
        adj[v2][u2] = 0
        
        adj[u][v2] = 1
        adj[v2][u] = 1
        adj[u2][v] = 1
        adj[v][u2] = 1
        
        # Update neighbors
        neighbors[u].remove(v)
        neighbors[u].append(v2)
        neighbors[v].remove(u)
        neighbors[v].append(u2) # Wait, v is connected to u2? No.
        # Swap (u,v) and (u2,v2) to (u,v2) and (u2,v)
        # u loses v, gains v2
        # v loses u, gains u2
        # u2 loses v2, gains v
        # v2 loses u2, gains u
        
        neighbors[v].remove(u)
        neighbors[v].append(u2)
        
        neighbors[u2].remove(v2)
        neighbors[u2].append(v)
        
        neighbors[v2].remove(u2)
        neighbors[v2].append(u)
        
        # Check girth
        new_girth = _calculate_girth_fast(adj, n)
        
        if new_girth > best_girth:
            best_girth = new_girth
            if best_girth >= g:
                return (n, _adj_to_edges(adj))
        else:
            # Revert
            adj[u][v] = 1
            adj[v][u] = 1
            adj[u2][v2] = 1
            adj[v2][u2] = 1
            
            adj[u][v2] = 0
            adj[v2][u] = 0
            adj[u2][v] = 0
            adj[v][u2] = 0
            
            # Revert neighbors
            neighbors[u].remove(v2)
            neighbors[u].append(v)
            neighbors[v].remove(u2)
            neighbors[v].append(u)
            neighbors[u2].remove(v)
            neighbors[u2].append(v2)
            neighbors[v2].remove(u)
            neighbors[v2].append(u2)

    return None

def _random_regular_graph(n, k):
    if n * k % 2 != 0:
        return None
    
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if adj[u][v] == 1:
            return None
        adj[u][v] = 1
        adj[v][u] = 1
        
    return adj

def _calculate_girth_fast(adj, n):
    min_girth = float('inf')
    for start in range(n):
        if not _time_ok(margin=3.0):
            break
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= min_girth:
                continue
            for v in range(n):
                if adj[u][v] == 1:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] > dist[u] + 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth <= 3:
                                return min_girth
                        
    return min_girth if min_girth != float('inf') else n + 1

def _adj_to_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
    return edges

def _simple_regular_graph(k):
    n = k + 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return (n, edges)

def _luw_construct(k, g, d_override=None):
    q = k
    if q not in (2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        return None

    if d_override is not None:
        d = d_override
    else:
        d = None
        for cand in range(3, 20):
            bound = cand + 5 if cand % 2 == 1 else cand + 4
            if bound >= g:
                d = cand
                break
        if d is None:
            d = max(3, g - 5)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q

    def encode(side, x):
        val = 0
        for c in range(1, d + 1):
            val += x[c] * q_powers[d - c]
        return side * (q ** d) + val

    def decode(enc):
        side = enc // (q ** d)
        val = enc % (q ** d)
        x = [0] * (d + 1)
        for c in range(1, d + 1):
            x[c] = val % q
            val //= q
        return side, x

    def neighbours(enc):
        side, x = decode(enc)
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield encode(1 - side, y)

    start_enc = encode(0, [0] * (d + 1))
    index = {start_enc: 0}
    order = [start_enc]
    edges = []

    for i, enc in enumerate(order):
        if not _time_ok(margin=15.0):
            return None
        side, _ = decode(enc)
        for other_enc in neighbours(enc):
            if other_enc not in index:
                index[other_enc] = len(order)
                order.append(other_enc)
            if side == 0:
                edges.append((index[enc], index[other_enc]))

    n = len(order)
    if n == 0:
        return None

    if _verify(n, edges, k, g):
        return (n, edges)
    return None

def _verify(n, edges, k, g):
    if n == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    for start in range(n):
        if not _time_ok(margin=10.0):
            return False
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1))
        
        while queue:
            v, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                new_state = (w, v)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if w == start and depth + 1 < g:
                    return False
                queue.append((w, v, depth + 1))
    
    return True
