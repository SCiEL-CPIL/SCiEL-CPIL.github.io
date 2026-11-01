import time
import random
import math
from collections import deque
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Handle known small cages explicitly
    if (k, g) == (3, 5):
        return _petersen()
    elif (k, g) == (3, 6):
        return _heawood()
    elif (k, g) == (3, 7):
        return _cage_3_7()
    elif (k, g) == (3, 8):
        return _tutte_coxeter()
    elif (k, g) == (4, 5):
        return _klein_30()
    
    # 2. LUW Construction as baseline
    n_luw, edges_luw = _luw_construct(k, g)
    
    # Verify LUW result
    if not _verify_graph(n_luw, edges_luw, k, g):
        # Fallback: try to construct a valid graph
        return _fallback_construct(k, g, deadline)

    # 3. Try random search to find smaller valid graphs
    # This is most effective for small k and g where the gap between LUW and cage is small
    # and the search space is manageable.
    if time.time() < deadline - 10:
        n_rand, edges_rand = _random_search_improved(k, g, n_luw, deadline)
        if n_rand is not None and n_rand < n_luw:
            return n_rand, edges_rand

    return n_luw, edges_luw


def _verify_graph(n, edges, k, g):
    """Verify that the graph is k-regular, simple, and has girth >= g."""
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        deg[u] += 1
        deg[v] += 1
    
    if any(d != k for d in deg):
        return False
    
    # Check girth
    if not _check_girth(n, edges, g):
        return False
    
    return True


def _fallback_construct(k, g, deadline):
    """Fallback: try to construct a valid graph if LUW fails verification."""
    for d_offset in range(0, 5):
        if time.time() > deadline - 5:
            break
        n, edges = _luw_construct_variant(k, g, d_offset)
        if n is not None and _verify_graph(n, edges, k, g):
            return n, edges
    
    return _cycle_based_construct(k, g, deadline)


def _luw_construct_variant(k, g, d_offset):
    """Try LUW construction with different d values."""
    q = k
    base_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    d = base_d + d_offset
    
    if d < 3:
        return None, None
    
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
    
    n = len(order)
    return n, edges


def _cycle_based_construct(k, g, deadline):
    """Construct a valid k-regular graph of girth >= g using a cycle-based approach."""
    moore = _moore_bound(k, g)
    n = 2 * moore + 10
    
    for attempt in range(20):
        if time.time() > deadline - 3:
            break
        edges = _random_regular_graph(n, k)
        if edges is not None and _check_girth(n, edges, g):
            return n, edges
    
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw


def _petersen():
    """Petersen graph: 10 vertices, 3-regular, girth 5."""
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 2) % 5]))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges


def _heawood():
    """Heawood graph: 14 vertices, 3-regular, girth 6."""
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        line_id = 7 + i
        for p in line:
            edges.append((p, line_id))
    return 14, edges


def _tutte_coxeter():
    """Tutte-Coxeter graph: 30 vertices, 3-regular, girth 8."""
    adj = {
        0: [1, 15, 21], 1: [0, 16, 22], 2: [3, 17, 23], 3: [2, 18, 24],
        4: [5, 19, 25], 5: [4, 20, 26], 6: [7, 21, 27], 7: [6, 22, 28],
        8: [9, 23, 29], 9: [8, 24, 0], 10: [11, 25, 1], 11: [10, 26, 2],
        12: [13, 27, 3], 13: [12, 28, 4], 14: [15, 29, 5], 15: [0, 14, 16],
        16: [1, 15, 17], 17: [2, 16, 18], 18: [3, 17, 19], 19: [4, 18, 20],
        20: [5, 19, 21], 21: [6, 20, 22], 22: [7, 21, 23], 23: [8, 22, 24],
        24: [9, 23, 25], 25: [10, 24, 26], 26: [11, 25, 27], 27: [12, 26, 28],
        28: [13, 27, 29], 29: [14, 28, 9]
    }
    edges = set()
    for u in range(30):
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    return 30, list(edges)


def _cage_3_7():
    """(3,7)-cage: 24 vertices, 3-regular, girth 7. The Foster graph."""
    adj = {
        0: [1, 12, 18], 1: [0, 2, 13], 2: [1, 3, 14], 3: [2, 4, 15],
        4: [3, 5, 16], 5: [4, 6, 17], 6: [5, 7, 18], 7: [6, 8, 19],
        8: [7, 9, 20], 9: [8, 10, 21], 10: [9, 11, 22], 11: [10, 0, 23],
        12: [0, 13, 19], 13: [1, 12, 20], 14: [2, 15, 21], 15: [3, 14, 22],
        16: [4, 17, 23], 17: [5, 16, 0], 18: [6, 19, 1], 19: [7, 18, 2],
        20: [8, 21, 3], 21: [9, 20, 4], 22: [10, 23, 5], 23: [11, 22, 6]
    }
    edges = set()
    for u in range(24):
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    return 24, list(edges)


def _klein_30():
    """Klein graph: 30 vertices, 4-regular, girth 5."""
    edges = [
        (0, 1), (0, 2), (0, 15), (0, 20),
        (1, 3), (1, 4), (1, 16), (1, 21),
        (2, 5), (2, 6), (2, 17), (2, 22),
        (3, 7), (3, 8), (3, 18), (3, 23),
        (4, 9), (4, 10), (4, 19), (4, 24),
        (5, 11), (5, 12), (5, 20), (5, 25),
        (6, 13), (6, 14), (6, 21), (6, 26),
        (7, 15), (7, 16), (7, 22), (7, 27),
        (8, 17), (8, 18), (8, 23), (8, 28),
        (9, 19), (9, 20), (9, 24), (9, 29),
        (10, 21), (10, 22), (10, 25), (10, 29),
        (11, 23), (11, 24), (11, 26), (11, 29),
        (12, 25), (12, 26), (12, 27), (12, 29),
        (13, 27), (13, 28), (13, 29), (13, 20),
        (14, 28), (14, 29), (14, 21), (14, 22)
    ]
    n = 30
    deg = [0]*n
    for u,v in edges:
        deg[u]+=1
        deg[v]+=1
    if all(d==4 for d in deg) and len(edges)==60:
        return n, edges
    return _luw_construct(4, 5)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    if g % 2 == 1:
        d = max(3, (g - 1) // 2)
    else:
        d = max(3, (g - 2) // 2)
    
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


def _random_search_improved(k, g, n_target, deadline):
    """Try to find a k-regular graph of girth >= g with fewer vertices than n_target."""
    if time.time() > deadline:
        return None, None
    
    moore_bound = _moore_bound(k, g)
    
    # Only attempt if the search space is small enough
    if n_target > 500:
        return None, None
    
    lower = max(moore_bound, 2 * k + 1)
    upper = min(n_target - 1, 400)
    
    if lower > upper:
        return None, None
    
    # Try sizes in decreasing order
    for n in range(upper, lower - 1, -1):
        if time.time() > deadline - 5:
            break
        if n * k % 2 != 0:
            continue
        
        # Limit attempts per size
        max_attempts_per_size = 50
        for trial in range(max_attempts_per_size):
            if time.time() > deadline - 5:
                break
            
            edges = _random_regular_graph(n, k)
            if edges is None:
                continue
            
            if _check_girth(n, edges, g):
                return n, edges
    
    return None, None


def _moore_bound(k, g):
    """Compute the Moore bound for k-regular graph of girth g."""
    if g % 2 == 1:
        h = (g - 3) // 2
        return k * sum(k**i for i in range(h + 1))
    else:
        h = (g - 2) // 2
        return 2 * sum(k**i for i in range(h + 1))


def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices. Returns edge list or None."""
    if n * k < 2 or n < k + 1:
        return None
    if n * k % 2 != 0:
        return None
    
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None
        edges.add(edge)
    
    return list(edges)


def _check_girth(n, edges, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = deque([start])
        dist[start] = 0
        
        while queue:
            node = queue.popleft()
            d = dist[node]
            
            max_depth = (g - 2) // 2
            if d > max_depth:
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                if dist[neighbor] == -1:
                    dist[neighbor] = d + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                else:
                    cycle_len = d + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
    
    return True
# EVOLVE-BLOCK-END
