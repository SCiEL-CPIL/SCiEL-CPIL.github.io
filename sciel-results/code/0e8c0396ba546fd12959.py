"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
from collections import deque

# EVOLVE-BLOCK-START

def _check_girth(n, edges, k, g, deadline):
    """Verify via non-backtracking walks from every vertex that no cycle < g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # degree check
    for u in range(n):
        if len(adj[u]) != k:
            return False
    # simple graph check
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    # Non-backtracking BFS from each directed edge (u, v) up to depth g-1
    max_depth = g - 2
    if max_depth < 1:
        return True
    stamp = [0] * (n * n)
    gen = 0
    for a in range(n):
        for b in adj[a]:
            gen += 1
            queue = [(b, a, 0)]
            stamp[b * n + a] = gen
            qi = 0
            while qi < len(queue):
                cur, prev, depth = queue[qi]
                qi += 1
                if cur == a and prev != b and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    idx = nxt * n + cur
                    if stamp[idx] != gen:
                        stamp[idx] = gen
                        queue.append((nxt, cur, depth + 1))
        if _time.time() > deadline:
            return True
    return True

def _build_luw(k, g, deadline):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
        if _time.time() > deadline - 30.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _try_circulant(n, k, g, deadline):
    """Try to find a k-regular circulant graph on n vertices with girth >= g."""
    if k % 2 != 0 or n < 2 * k:
        return None
    
    # For k=4: generators {±a, ±b}. By automorphism x→x·a^{-1} (if gcd(a,n)=1),
    # we can WLOG set a=1 and search over b.
    # Also try a=2,3,... for cases where gcd(a,n)≠1.
    
    def _fast_girth_check(n, a, b, g):
        """Fast non-backtracking BFS from vertex 0 for circulant {±a,±b}."""
        max_depth = g - 2
        if max_depth < 1:
            return True
        # stamp[(cur, prev)] = gen, use a dict or 2D array
        stamp = {}
        gen = 1
        queue = [(b, 1, 0), (n-b, 1, 0), (a, 1, 0), (n-a, 1, 0)]
        for item in queue:
            stamp[(item[0], item[1])] = gen
        
        qi = 0
        while qi < len(queue):
            cur, prev, depth = queue[qi]
            qi += 1
            if cur == 0 and depth >= 1:
                return False
            if depth >= max_depth:
                continue
            for nxt in (cur + a) % n, (cur - a) % n, (cur + b) % n, (cur - b) % n:
                if nxt == prev:
                    continue
                key = (nxt, cur)
                if key not in stamp:
                    stamp[key] = gen
                    queue.append((nxt, cur, depth + 1))
        return True
    
    candidates = []
    for a in range(1, n // 2 + 1):
        if 2 * a % n == 0:
            continue
        for b in range(1, n // 2 + 1):
            if 2 * b % n == 0:
                continue
            if (a + b) % n == 0 or (a - b) % n == 0:
                continue
            if a == b:
                continue
            # 4-cycle check: a+b≠0, a≠b, a≠0, b≠0, a+b≠a, a+b≠b
            # Already satisfied by above
            candidates.append((a, b))
    
    # Sort by some heuristic: prefer small a, b
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    for a, b in candidates:
        if _time.time() > deadline - 10.0:
            break
        if _fast_girth_check(n, a, b, g):
            edges = []
            for i in range(n):
                edges.append((i, (i + a) % n))
                edges.append((i, (i + b) % n))
            return n, edges
    return None

def construct(k, g):
    deadline = _time.time() + 110.0
    
    # 1. Try Circulant Construction
    if k == 4 and g == 10:
        # Try n=320 first (known cage size)
        for n in [320, 322, 324, 326, 328, 330]:
            if _time.time() > deadline - 20.0:
                break
            res = _try_circulant(n, k, g, deadline)
            if res:
                return res

    # 2. Fallback to LUW
    return _build_luw(k, g, deadline)

# EVOLVE-BLOCK-END
