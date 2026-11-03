import math
import time
import random
import sys
from collections import deque

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)

    start_time = time.time()
    deadline = 112  # hard safety margin before 150s kill

    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1

    best = None  # (n, edges)

    # ---- 1. Generalized Petersen graph search (structural, fast) ----
    # GP(n, k) has 2n vertices. Girth depends on n and k.
    # For girth >= g, we need specific (n, k) pairs.
    gp_result = _search_gp_graphs(g, start_time, deadline)
    if gp_result is not None:
        best = gp_result

    # ---- 2. Circulant C(n; {a, n-a, n/2}) scan ----
    max_circ_n = 1400
    for n in range(moore_bound, max_circ_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        if best is not None and n >= best[0]:
            continue
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant(n, a, g, start_time, deadline):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                if best is None or n < best[0]:
                    best = (n, edges)
                break

    # ---- 3. Random 3-regular search ----
    max_random_n = 1400
    for n in range(moore_bound, max_random_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        if best is not None and n >= best[0]:
            continue
        attempts = 3 if n < 800 else (6 if n < 1200 else 12)
        for _ in range(attempts):
            if time.time() - start_time > deadline:
                break
            edges = _generate_random_3_reg(n)
            if edges is None:
                continue
            if not _is_connected(n, edges):
                continue
            if _check_girth_general(n, edges, g, start_time, deadline):
                if best is None or n < best[0]:
                    best = (n, edges)
                break

    # ---- 4. LUW fallback ----
    if best is None:
        n, edges = _luw_fallback(k, g)
        best = (n, edges)
    else:
        n, edges = _luw_fallback(k, g)
        if n < best[0]:
            best = (n, edges)

    return best


def _search_gp_graphs(g, start_time, deadline):
    """Search generalized Petersen graphs GP(n, k) for girth >= g."""
    # GP(n, k): vertices 0..2n-1. Outer cycle: i-(i+1) mod n.
    # Spokes: i-(i+k) mod n (for i in 0..n-1, connecting outer i to inner i+k)
    # Actually GP(n,k): outer vertices 0..n-1 in a cycle, inner vertices n..2n-1
    # Outer: i connected to (i-1)%n, (i+1)%n, and i+k (mod n) mapped to n+(i+k)%n
    # Inner: n+i connected to n+(i-1)%n, n+(i+1)%n, and i (mod n)
    
    # For girth 16, known small GP graphs:
    # We need to check many (n, k) pairs.
    best = None
    max_n = 500  # 2n vertices, so up to 1000 vertices
    
    for n in range(10, max_n + 1):
        if time.time() - start_time > deadline:
            break
        if 2 * n >= 936:
            break
        for k in range(1, n // 2):
            if math.gcd(n, k) != 1:
                continue
            # Quick check: build edges and verify girth
            edges = _gp_edges(n, k)
            if _check_girth_general(2 * n, edges, g, start_time, deadline):
                if best is None or 2 * n < best[0]:
                    best = (2 * n, edges)
                break  # for this n, found a valid k; move to next n
    
    return best


def _gp_edges(n, k):
    """Build edge list for generalized Petersen graph GP(n, k)."""
    edges = []
    seen = set()
    # Outer cycle: i - (i+1) % n
    for i in range(n):
        u, v = i, (i + 1) % n
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    # Inner cycle: n+i - n+(i+1) % n
    for i in range(n):
        u, v = n + i, n + (i + 1) % n
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    # Spokes: i - n+(i+k) % n
    for i in range(n):
        u, v = i, n + (i + k) % n
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    return edges


def _check_girth_circulant(n, a, g, start_time, deadline):
    """Non-backtracking walk check for C(n; {a, -a, n/2}) using NumPy."""
    h = n // 2
    S = [a % n, (n - a) % n, h % n]
    neg_idx = [1, 0, 2]

    R = np.zeros((3, n), dtype=bool)
    R[0, S[0]] = True
    R[1, S[1]] = True
    R[2, S[2]] = True

    for d in range(2, g):
        if time.time() - start_time > deadline:
            return False
        R_new = np.zeros((3, n), dtype=bool)
        for i in range(3):
            for j in range(3):
                if j == neg_idx[i]:
                    continue
                R_new[j] |= np.roll(R[i], S[j])
        R = R_new
        if R[:, 0].any():
            return False

    return True


def _generate_random_3_reg(n):
    """Random simple 3-regular graph on n vertices (configuration model)."""
    if n % 2 != 0 or n < 4:
        return None
    for _ in range(100):
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        valid = True
        edge_set = set()
        for i in range(0, 3 * n, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            edges.append((u, v))
        if valid:
            return edges
    return None


def _is_connected(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = [False] * n
    q = deque([0])
    visited[0] = True
    count = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
    return count == n


def _check_girth_general(n, edges, g, start_time, deadline):
    """BFS from every vertex, detect any cycle of length < g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        if time.time() - start_time > deadline:
            return False
        visited = {start: 0}
        q = deque([(start, -1, 0)])
        while q:
            u, parent, d = q.popleft()
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    dv = visited[v]
                    if d + dv + 1 < g:
                        return False
                else:
                    visited[v] = d
                    q.append((v, u, d + 1))
    return True


def _luw_fallback(k, g):
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
