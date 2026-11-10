"""Search for a small 3-regular graph of girth >= g, falling back to LUW construction."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
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


def _girth_at_least_nb(adj, n, g, t0=None, deadline=140.0):
    """Return True iff graph (adjacency list) has girth >= g."""
    if g <= 2:
        return True
    
    max_depth = g - 1
    from collections import deque
    
    for s in range(n):
        visited = set()
        queue = deque()
        for u in adj[s]:
            state = (u, s)
            visited.add(state)
            queue.append((u, s, 1))
        
        while queue:
            v, p, depth = queue.popleft()
            
            if t0 is not None and (time.time() - t0) > deadline:
                return False
            
            if depth >= max_depth:
                continue
            
            nd = depth + 1
            for u in adj[v]:
                if u == p:
                    continue
                if u == s:
                    if nd < g:
                        return False
                    continue
                state = (u, v)
                if state not in visited:
                    visited.add(state)
                    queue.append((u, v, nd))
    return True


def _random_regular_graph(n, k, rng):
    """Generate a random simple k-regular graph on n vertices (n*k even).
    Returns adjacency list or None on failure."""
    if n * k % 2 != 0 or n < k + 1:
        return None
    # configuration model
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    for _attempt in range(200):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, n * k, 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok and all(len(x) == k for x in adj):
            return [sorted(x) for x in adj]
    return None


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0  # safety margin under 150s wall-clock

    # Moore bound lower bound on n
    if g >= 4:
        # For k-regular, girth g even (g=2r):
        # n >= 2 * sum_{i=0}^{r-1} (k-1)^i
        # For k=3: n >= 2 * (2^r - 1)
        r = g // 2
        moore = 2 * ((k - 1) ** r - 1)
        # Ensure it's an integer and at least k+1
        if moore < k + 1:
            moore = k + 1
    else:
        moore = 0

    rng = random.Random(12345)

    # Build a list of candidate n values: even numbers (n*k must be even -> n even for k odd)
    step = 2 if k % 2 == 1 else 1
    lo = max(moore, k + 1)
    if lo % step:
        lo += step
    
    # Search window: try from Moore bound upward
    # Use the fast non-backtracking checker to enable more attempts
    hi = lo + 600  # wider search window with faster checker

    n = lo
    while n <= hi and not _deadline_reached(t0, HARD_LIMIT):
        attempts_per_n = 0
        # With the faster checker, we can afford more attempts
        max_attempts = 30 if n < 800 else 10
        while attempts_per_n < max_attempts and not _deadline_reached(t0, HARD_LIMIT):
            adj = _random_regular_graph(n, k, rng)
            attempts_per_n += 1
            if adj is None:
                continue
            if _girth_at_least_nb(adj, n, g, t0=t0, deadline=HARD_LIMIT):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        n += step

    # Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        raise
# EVOLVE-BLOCK-END
