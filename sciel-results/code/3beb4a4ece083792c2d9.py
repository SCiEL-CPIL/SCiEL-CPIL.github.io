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
            # Tutte-Coxeter (3,7)-cage, 30 vertices
            # This is the (3,7)-cage
            edges = []
            # Incidence graph of the (3,7)-configuration
            # 15 points, 15 lines
            # Points: 0..14, Lines: 15..29
            # This is a specific graph. Let's use a known construction or just return a valid one if we can generate it quickly.
            # Actually, for k=3, g=7, the Tutte-Coxeter graph has 30 vertices.
            # We can construct it from the (3,7)-config.
            # For now, let's fall through to LUW which should find a good one, or hardcode if easy.
            pass
    
    # For k=5, g=9, try to find a graph smaller than LUW (1560)
    if k == 5 and g == 9:
        # The best known is 1116 (Exoo et al. 2026).
        # LUW CD(7, 5) has 2 * 5^7 = 156250 vertices? No, the formula is different.
        # The LUW graph CD(d, q) has 2 * q^d vertices? No.
        # The Lazebnik-Ustimenko-Woldar graph CD(d, q) has 2 * q^d vertices.
        # Wait, the standard LUW graph has 2 * q^d vertices?
        # Let's check the number of vertices for CD(5, 5) or similar.
        # The girth of CD(d, q) is at least d+1 (if d is odd) or d+2?
        # Actually, the girth of the LUW graph CD(d, q) is 2*floor(d/2) + 2? No.
        # The LUW graph is a bipartite graph with parts of size q^d.
        # So n = 2 * q^d.
        # For k=5, q=5.
        # If d=5, n = 2 * 5^5 = 31250. Too big.
        # If d=4, n = 2 * 5^4 = 12500.
        # The girth of CD(d, q) is at least d+1?
        # Actually, the girth is known to be at least 2*floor((d+1)/2) + 2?
        # Let's just try to find a graph using a different method or accept LUW if it's small enough.
        # But 1116 is much smaller.
        # Maybe we can use a known 1116 vertex graph?
        # Or we can try a greedy search with a better heuristic.
        
        # Let's try to find a graph with n close to the Moore bound.
        # Moore bound for k=5, g=9:
        # For g=9 (odd), M(5, 9) = 5 + 5 * sum_{i=0}^{3} (5-1)^i = 5 + 5 * (1 + 4 + 16 + 64) = 5 + 5 * 85 = 430.
        # So n >= 430.
        # The best known is 1116.
        # LUW gives 2 * q^d. If we pick q=5, d=5, n=31250. Too big.
        # Maybe there are other constructions.
        
        # Let's try a greedy search for n in [1116, 1560].
        # The previous code tried this but maybe it was too slow or didn't find one.
        # Let's improve the greedy search.
        
        best_result = None
        # Try a few random restarts
        for n in [1116, 1120, 1124, 1130]:
            if not _time_ok(margin=10.0):
                break
            result = _greedy_girth_max_v2(n, k, g)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            return best_result
        
        # Fallback to LUW
        return _luw_construct(k, g)

    # General fallback: LUW
    return _luw_construct(k, g)

def _greedy_girth_max_v2(n, k, g):
    """Try to construct a k-regular graph on n vertices with girth >= g using greedy search."""
    if not _time_ok(margin=5.0):
        return None
    
    # Generate random regular graph
    adj = _random_regular_graph(n, k)
    if adj is None:
        return None
    
    # Initial girth check
    girth = _calculate_girth_fast(adj, n)
    if girth >= g:
        return (n, _adj_to_edges(adj))
    
    # Greedy improvement
    best_girth = girth
    best_adj = [row.copy() for row in adj]
    
    # Limit iterations
    max_iters = 10000
    for _ in range(max_iters):
        if not _time_ok(margin=5.0):
            break
        
        # Randomly pick two edges to swap
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or adj[u][v] == 0:
            continue
        
        u2 = random.randint(0, n - 1)
        v2 = random.randint(0, n - 1)
        if u2 == v2 or adj[u2][v2] == 0:
            continue
        
        # Ensure we don't create multi-edges or loops
        if adj[u][v2] != 0 or adj[u2][v] != 0:
            continue
        if u == v2 or u2 == v:
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
        
        # Check girth
        new_girth = _calculate_girth_fast(adj, n)
        
        if new_girth > best_girth:
            best_girth = new_girth
            best_adj = [row.copy() for row in adj]
            if best_girth >= g:
                return (n, _adj_to_edges(best_adj))
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

    return None

def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    # Pair stubs
    adj = np.zeros((n, n), dtype=int)
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
    """Calculate girth using BFS from every vertex."""
    min_girth = float('inf')
    
    for start in range(n):
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
                            
    return min_girth if min_girth != float('inf') else n + 1

def _adj_to_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
    return edges

def _luw_construct(k, g, d_override=None):
    """Build Lazebnik-Ustimenko-Woldar graph CD(d, q) with q=k."""
    q = k
    if q not in (4, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
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
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
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
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] + [0] * d
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        if not _time_ok(margin=15.0):
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

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
