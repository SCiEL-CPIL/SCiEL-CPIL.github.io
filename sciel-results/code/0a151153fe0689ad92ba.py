import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # ---------- LUW fallback (guaranteed valid 3-regular, high girth) ----------
    def luw_construct(k, g):
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

    # ---------- Fast certified girth check (non-backtracking, per-vertex early exit) ----------
    def has_short_cycle(adj, n, g):
        # True iff a cycle of length < g exists.
        # A cycle of length L exists iff some vertex is reached by a non-backtracking
        # walk of length L (L>=3) that returns to it.
        # Non-backtracking walk state: (current vertex, previous vertex).
        for start in range(n):
            # state -> step count
            seen = {}
            for nb in adj[start]:
                seen[(nb, start)] = 1
            frontier = [(nb, start) for nb in adj[start]]
            dist = 1
            while frontier and dist < g:
                dist += 1
                new_frontier = []
                for (curr, prev) in frontier:
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        if nb == start:
                            # non-backtracking walk of length dist returning to start
                            return True
                        state = (nb, curr)
                        if state not in seen:
                            seen[state] = dist
                            new_frontier.append(state)
                frontier = new_frontier
        return False

    # ---------- Petersen base graph ----------
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),   # outer 5-cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),   # inner 5-cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)    # spokes
    ]
    n_base = 10
    m_base = len(base_edges)  # 15
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)

    # ---------- Build voltage graph (Z_m) and return (n, edges) or None if not simple ----------
    def build_voltage(volts, m):
        # volts: list of length m_base, volt on each base edge (group element in Z_m)
        n = n_base * m
        edges = []
        for i in range(n_base):
            for j in base_adj[i]:
                if i >= j:
                    continue
                # find edge index
                # precompute map
        # Precompute edge index map
        # (recompute cheaply)
        edge_index = {}
        for idx, (u, v) in enumerate(base_edges):
            edge_index[(min(u, v), max(u, v))] = idx
        for i in range(n_base):
            for j in base_adj[i]:
                if i >= j:
                    continue
                eidx = edge_index[(i, j)]
                volt = volts[eidx]
                for coset in range(m):
                    u = i * m + coset
                    v = j * m + (coset + volt) % m
                    edges.append((u, v))
        # Verify simplicity: no loops, no repeated edges
        if len(edges) != 3 * n // 2:
            return None
        seen = set()
        for (u, v) in edges:
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in seen:
                return None
            seen.add(key)
        # Verify degree 3
        deg = [0] * n
        for (u, v) in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != k:
                return None
        return n, edges

    # ---------- LUW baseline ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # ---------- Voltage search over prime groups ----------
    def is_prime(num):
        if num < 2:
            return False
        if num < 4:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

    # Moore bound for k=3, g=16 is 512, so m >= 52 -> candidate_n = 10*m >= 512
    # Search m in descending order so we find the smallest n first.
    # n = 10*m, so smaller m -> smaller n.
    # We want n < best_n, so m < best_n / 10.
    # Start from the smallest feasible m upward, but we want the BEST (smallest) n.
    # Search m from low to high, return the first valid one found (smallest n).
    # But we must also beat best_n.
    
    # Feasible m: 10*m >= 512 (Moore) and 10*m < best_n
    m_min = 52
    m_max = best_n // 10 - 1  # so that 10*m < best_n
    if m_max < m_min:
        # Can't beat LUW with voltage over Petersen; return LUW
        return best_n, best_edges

    # Search m from low to high (smallest n first). Return first valid.
    # But to maximize chance, search in order of increasing m.
    # Use a deadline-aware loop.
    rng = random.Random(12345)
    
    # Precompute edge index map once
    edge_index = {}
    for idx, (u, v) in enumerate(base_edges):
        edge_index[(min(u, v), max(u, v))] = idx
    # For each vertex, list of (neighbor, edge_index)
    base_adj_ei = [[] for _ in range(n_base)]
    for idx, (u, v) in enumerate(base_edges):
        base_adj_ei[u].append((v, idx))
        base_adj_ei[v].append((u, idx))

    for m in range(m_min, m_max + 1):
        if time.time() > deadline - 5.0:
            break
        if not is_prime(m):
            continue
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue

        # Try several voltage assignments for this m.
        # Number of trials scales with remaining time.
        remaining = deadline - 5.0 - time.time()
        # Each trial: build (O(n)) + girth check (O(n * 3^(g/2)) worst, but early exit).
        # For g=16, girth check per vertex explores up to 3^7 = 2187 states before depth 8.
        # n up to ~900 -> ~2e6 state visits, ~0.1-0.3s in Python. Too slow for many trials.
        # So keep trials modest.
        trials = max(2, int(remaining * 3))
        trials = min(trials, 60)

        found = False
        for t in range(trials):
            if time.time() > deadline - 3.0:
                break
            # Random voltage assignment: each edge gets a random non-zero element of Z_m.
            # Non-zero elements are 1..m-1 (all generators since m prime).
            volts = [rng.randint(1, m - 1) for _ in range(m_base)]
            result = build_voltage(volts, m)
            if result is None:
                continue
            n, edges = result
            # Build adjacency for girth check
            adj = [[] for _ in range(n)]
            for (u, v) in edges:
                adj[u].append(v)
                adj[v].append(u)
            if not has_short_cycle(adj, n, g):
                return n, edges

    return best_n, best_edges
