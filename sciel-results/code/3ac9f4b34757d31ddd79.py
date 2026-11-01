"""
Construct a k-regular graph with girth at least g.
Structural change: randomized circulant search with O(1) amortized per candidate.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        result = _find_circulant_random(3, 16, deadline)
        if result is not None:
            return result

        for d in [9, 11, 13, 15]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _find_circulant_random(k, g, deadline):
    """
    Randomized search for C_n(1,a,b) with girth >= g.
    For each n, try a fixed number of random (a,b) pairs.
    """
    min_n = 512
    max_n = 940
    trials_per_n = 50  # random trials per n

    rng = random.Random(42)

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        half = n // 2
        tried = set()

        for _ in range(trials_per_n):
            if time.time() > deadline - 5:
                return None

            # Random a, b with a < b, both in [2, half)
            a = rng.randint(2, half - 1)
            b = rng.randint(a + 1, half - 1)

            if (a, b) in tried:
                continue
            tried.add((a, b))

            # Quick filters
            if _has_4cycle(n, a, b):
                continue
            if _has_6cycle(n, a, b):
                continue

            # Full girth check
            if _girth_check_fast(n, a, b, g):
                edges = []
                for i in range(n):
                    for shift in [1, a, b]:
                        j = (i + shift) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    return None


def _has_4cycle(n, a, b):
    G = [1, n - 1, a, n - a, b, n - b]
    for i in range(6):
        for j in range(i + 1, 6):
            s = (G[i] + G[j]) % n
            for k in range(6):
                if k != i and k != j and G[k] == s:
                    return True
    return False


def _has_6cycle(n, a, b):
    G = [1, n - 1, a, n - a, b, n - b]
    for i in range(6):
        for j in range(6):
            for k in range(6):
                if G[i] == G[j] or G[j] == G[k] or G[i] == G[k]:
                    continue
                if (n - G[i]) % n == G[j] or (n - G[j]) % n == G[k] or (n - G[i]) % n == G[k]:
                    continue
                if (G[i] + G[j] + G[k]) % n == 0:
                    return True
    return False


def _girth_check_fast(n, a, b, g):
    """
    Check girth >= g for C_n(1,a,b) using non-backtracking walk BFS from vertex 0.
    """
    steps = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    # Deduplicate
    seen = set()
    unique_steps = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            unique_steps.append(s)
    steps = unique_steps

    inv_map = {}
    for s in steps:
        inv_map[s] = (n - s) % n

    visited = set()
    queue = deque()

    for s in steps:
        v = s % n
        if v == 0:
            continue
        state = (v, s)
        visited.add(state)
        queue.append((state, 1))

    max_len = g - 1

    while queue:
        (v, last_s), d = queue.popleft()
        if d >= max_len:
            continue
        for s in steps:
            if inv_map[last_s] == s:
                continue
            new_v = (v + s) % n
            new_d = d + 1
            if new_d > max_len:
                continue
            if new_v == 0 and new_d >= 3:
                return False
            new_state = (new_v, s)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, new_d))

    return True


def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    q = k
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
                aa, bb = rule[c]
                y[c] = add(x[c], mul(y[aa], x[bb])) if side == 0 else sub(x[c], mul(x[aa], y[bb]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False
    return True


def _bfs_short_cycle(src, adj, n, g):
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    max_depth = g // 2
    while queue:
        u = queue.popleft()
        depth = dist[u]
        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    if depth + 1 + dist[v] < g:
                        return True
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                if depth + 1 + dist[v] < g:
                    return True
    return False

# EVOLVE-BLOCK-END
