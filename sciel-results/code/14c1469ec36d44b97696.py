import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin vs 150s kill
    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)


# ------------------- Petersen voltage-lift method -------------------

def _construct_petersen_lift(g, deadline, start_time):
    # Petersen graph: 10 vertices, 15 edges
    # Outer cycle 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes 0-5..4-9
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    n_base = 10
    n_edges = len(pet_edges)

    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Closed non-backtracking walks of length 3..g-1, as sparse coefficient vectors
    def _closed_walk_coeffs(max_len):
        walks = set()
        for start_v in range(n_base):
            stack = []
            for next_v, e_id in base_adj[start_v]:
                u, v = pet_edges[e_id]
                sign = 1 if u == start_v else -1
                stack.append((next_v, e_id, 1, [(e_id, sign)]))
            while stack:
                curr_v, prev_e, length, path = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = pet_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    stack.append((next_v, e_id, length + 1, path + [(e_id, sign)]))

        sparse = []
        seen = set()
        for w in walks:
            coeff = [0] * n_edges
            for e_id, sign in w:
                coeff[e_id] += sign
            nz = [(e, c) for e, c in enumerate(coeff) if c != 0]
            if not nz:
                continue
            key = tuple(sorted(nz))
            if key not in seen:
                seen.add(key)
                sparse.append(nz)
        return sparse

    walk_coeffs = _closed_walk_coeffs(g)

    def _is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _valid(volts, m):
        # A walk's voltage sum is 0 mod m  <=>  divisible by m.
        # For m >= 128 the short-walk sum is tiny -> only 0 is possible.
        if m >= 128:
            for nz in walk_coeffs:
                s = 0
                for e, c in nz:
                    s += c * volts[e]
                if s == 0:
                    return False
            return True
        if _is_prime(m):
            for nz in walk_coeffs:
                s = 0
                for e, c in nz:
                    s += c * volts[e]
                if s == 0:
                    return False
            return True
        for nz in walk_coeffs:
            s = 0
            for e, c in nz:
                s += c * volts[e]
            if s % m == 0:
                return False
        return True

    def _build_lift(volts, m):
        edges = []
        for i, (u, v) in enumerate(pet_edges):
            vol = volts[i]
            for t in range(m):
                a = u * m + t
                b = v * m + (t + vol) % m
                edges.append((a, b))
        return edges

    def _verify(n, edges, girth_req, k):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for start in range(n):
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= girth_req:
                    continue
                if curr == start and dist >= 3:
                    return False
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        queue.append((nb, curr, dist + 1))
        return True

    # Candidate moduli (n = 10*m). Prefer coprime-voltage-friendly / prime m.
    m_candidates = []
    for m in range(52, 200):
        if _is_prime(m):
            m_candidates.append(m)
    for m in range(200, 300, 2):
        if _is_prime(m):
            m_candidates.append(m)
    # add a few composite coprime-friendly moduli (2^a * 3^b)
    for m in (64, 96, 128, 192, 256, 384):
        if m not in m_candidates:
            m_candidates.append(m)
    m_candidates = sorted(set(m_candidates))

    random.seed(12345)

    best_n = float('inf')
    best_edges = None

    for m in m_candidates:
        if time.time() > deadline - 25.0:
            break
        cand_n = n_base * m
        if cand_n >= best_n:
            continue

        max_tries = 150000 if m < 80 else 60000
        found = False
        for _ in range(max_tries):
            if time.time() > deadline - 12.0:
                break
            volts = [random.randint(1, m - 1) for _ in range(n_edges)]
            # gcd prefilter: gcd of all voltages and m must be 1 (else a proper
            # quotient cover exists and short walks survive)
            gv = m
            for v in volts:
                gv = math.gcd(gv, v)
                if gv == 1:
                    break
            if gv != 1:
                continue
            if _valid(volts, m):
                edges = _build_lift(volts, m)
                if _verify(cand_n, edges, g, 3):
                    best_n = cand_n
                    best_edges = edges
                    found = True
                    break
        if found:
            break

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: LUW
    return _construct_luw(k=3, g=g, d=11)


# ------------------- LUW fallback -------------------

def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
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
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
