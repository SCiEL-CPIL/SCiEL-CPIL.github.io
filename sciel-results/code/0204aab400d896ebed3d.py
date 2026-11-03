"""Construct a 3-regular graph with girth at least 16."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Moore bound for (k, g): for even g=2r, MB = 2*(k-1)^(r-1)
    # For k=3, g=16, r=8: MB = 2*2^7 = 256
    # We'll aim for graphs near this bound
    
    # Try LUW construction first (deterministic, fast)
    result = _luw_construct(k, g, deadline)
    if result is not None:
        return result
    
    # Fallback: random regular graph search
    result = _random_search(k, g, deadline)
    if result is not None:
        return result
    
    # Last resort: return LUW even if large
    return _luw_construct(k, g, deadline)


def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
    q = k
    # For q-regular, need girth >= d+5 (odd d) or d+4 (even d)
    # To get girth >= g: if g is even, need d+5 >= g -> d >= g-5 (odd)
    # if g is odd, need d+5 >= g -> d >= g-5 (odd) or d+4 >= g -> d >= g-4 (even)
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            return None
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    n = len(order)
    return n, edges


def _verify_girth(n, edges, g):
    """Verify girth >= g using non-backtracking walks from each vertex."""
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking: state is (vertex, prev_vertex)
    # BFS from each vertex
    for start in range(n):
        # State: (current, prev), distance
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        
        while queue:
            if time.time() > time.time() + 1:  # simple guard
                continue
            curr, prev, dist = queue.popleft()
            if dist >= g:
                break  # found cycle of length >= g, continue checking others
            for nb in adj[curr]:
                if nb == prev:
                    continue  # non-backtracking
                if (nb, curr) in visited:
                    if dist + 1 < g:
                        return False  # found cycle shorter than g
                else:
                    visited[(nb, curr)] = dist + 1
                    queue.append((nb, curr, dist + 1))
    
    return True


def _random_search(k, g, deadline):
    """Try to find a small 3-regular graph with girth >= g via random search."""
    # Moore bound
    r = g // 2
    moore_bound = 2 * (k - 1) ** (r - 1)
    
    # Try sizes from Moore bound upward
    for n in range(moore_bound, moore_bound + 50, 2):  # must be even for 3-regular
        if n * k % 2 != 0:
            continue
        if time.time() > deadline - 10:
            break
        
        # Try to construct a random 3-regular graph
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            edges = _random_regular_graph(n, k)
            if edges is None:
                continue
            # Verify girth
            if _verify_girth(n, edges, g):
                return n, edges
    
    return None


def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    random.seed(42 + n + k)  # deterministic for reproducibility
    # Each vertex has k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # self-loop
        if (min(u, v), max(u, v)) in edges:
            return None  # multi-edge
        edges.add((min(u, v), max(u, v)))
    
    return list(edges)


# EVOLVE-BLOCK-END
