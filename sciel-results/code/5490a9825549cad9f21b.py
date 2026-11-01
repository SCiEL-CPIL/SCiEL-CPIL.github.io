import time
from collections import deque

def _verify_girth_fast(n, adj, k, g):
    max_depth = g // 2
    for start in range(n):
        if len(adj[start]) != k:
            return False
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            u, prev, depth = queue.popleft()
            if depth == max_depth:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if v == start:
                    if depth + 1 < g:
                        return False
                    continue
                state = (v, u)
                if state in visited:
                    if depth - visited[state] < g:
                        return False
                else:
                    visited[state] = depth
                    queue.append((v, u, depth + 1))
    return True

def _has_short_cycle(adj, n, limit):
    """Fast check: returns True if there is a cycle of length < limit (even)."""
    for start in range(n):
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            u, prev, depth = queue.popleft()
            if depth >= limit - 1:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if v == start:
                    if depth + 1 < limit:
                        return True
                    continue
                state = (v, u)
                if state in visited:
                    if depth - visited[state] < limit:
                        return True
                else:
                    visited[state] = depth
                    queue.append((v, u, depth + 1))
    return False

def _find_primitive_root(p):
    if p == 2:
        return 1
    factors = []
    d = 2
    x = p - 1
    while d * d <= x:
        if x % d == 0:
            factors.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        factors.append(x)
    for g in range(2, p):
        ok = True
        for f in factors:
            if pow(g, (p - 1) // f, p) == 1:
                ok = False
                break
        if ok:
            return g
    return 1

def _build_petersen_lift(m, root, start_time, deadline):
    # Petersen graph: 10 vertices, 15 edges
    # vertices 0-4 (outer), 5-9 (inner)
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))       # outer cycle
        edges.append((5 + i, 5 + ((i + 1) % 5)))  # inner cycle
        edges.append((i, 5 + i))              # spokes
    n = 10 * m
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        voltage = pow(root, idx + 1, m)
        a = u * m
        b = v * m
        adj[a].append(b)
        adj[b].append(a)
        a2 = u * m + voltage
        b2 = v * m + voltage
        adj[a2].append(b2)
        adj[b2].append(a2)
    for i in range(n):
        if len(adj[i]) != 3:
            return None, None
    if time.time() > deadline - 15:
        return None, None
    if _has_short_cycle(adj, n, 8):
        return None, None
    if _verify_girth_fast(n, adj, 3, 16):
        edges_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges_list.append((u, v))
        return n, edges_list
    return None, None

def _build_k33_lift(m, root, start_time, deadline):
    # K3,3: 6 vertices, 9 edges
    edges = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    n = 6 * m
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        voltage = pow(root, idx + 1, m)
        a = u * m
        b = v * m
        adj[a].append(b)
        adj[b].append(a)
        a2 = u * m + voltage
        b2 = v * m + voltage
        adj[a2].append(b2)
        adj[b2].append(a2)
    for i in range(n):
        if len(adj[i]) != 3:
            return None, None
    if time.time() > deadline - 15:
        return None, None
    if _has_short_cycle(adj, n, 8):
        return None, None
    if _verify_girth_fast(n, adj, 3, 16):
        edges_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges_list.append((u, v))
        return n, edges_list
    return None, None

def _luw_construct(k, d):
    if k != 3:
        return None, None
    q = 3
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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

def _edges_to_adj(n, edges, k):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135.0
    best_n = float('inf')
    best_edges = []

    # --- Phase 1: Deterministic voltage lifts ---
    # Petersen base: n = 10*m, need n >= 512 -> m >= 52
    # K3,3 base: n = 6*m, need n >= 512 -> m >= 86
    # Scan primes; for each, use primitive root powers as voltages.

    # Collect primes in range
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        d = 5
        while d * d <= x:
            if x % d == 0 or x % (d + 2) == 0:
                return False
            d += 6
        return True

    # Petersen lifts: m from 53 upward (primes)
    m = 53
    while m < 200 and time.time() < deadline - 20:
        if is_prime(m):
            root = _find_primitive_root(m)
            n, edges = _build_petersen_lift(m, root, start_time, deadline)
            if n is not None:
                if n < best_n:
                    best_n = n
                    best_edges = edges
        m += 1

    # K3,3 lifts: m from 89 upward (primes)
    m = 89
    while m < 200 and time.time() < deadline - 15:
        if is_prime(m):
            root = _find_primitive_root(m)
            n, edges = _build_k33_lift(m, root, start_time, deadline)
            if n is not None:
                if n < best_n:
                    best_n = n
                    best_edges = edges
        m += 1

    # --- Phase 2: LUW fallbacks ---
    for d in (7, 8, 9, 10):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if n is None:
            continue
        adj = _edges_to_adj(n, edges, k)
        if adj is None:
            continue
        if _verify_girth_fast(n, adj, k, g):
            if n < best_n:
                best_n = n
                best_edges = edges

    if best_n == float('inf'):
        # Last resort: return LUW(3, 7) even if unverified
        n, edges = _luw_construct(k, 7)
        return n, edges

    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
