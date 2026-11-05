import time
import random
import math
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Primary strategy: voltage lift of the Petersen graph (k=3, g=16) using a
    STRUCTURED search (base assignment + small perturbations) with a
    gcd-divisibility pre-filter and a vectorized constraint check, to find a
    voltage assignment that kills all short non-backtracking closed walks.
    This targets n = 10*m, far smaller than LUW constructions.
    Fallback: LUW construction.
    """
    start_time = time.time()
    # Evaluator kills at 150s. Keep a safety margin.
    deadline = start_time + 135.0

    if k == 3 and g == 16:
        result = _construct_voltage_lift(g, deadline)
        if result is not None:
            return result
        return _construct_luw(k, g, d=11)
    elif k == 3:
        if g >= 8:
            result = _construct_voltage_lift(g, deadline)
            if result is not None:
                return result
        return _construct_luw(k, g, d=max(3, (g - 1) // 2))
    else:
        return _construct_luw(k, g, d=7)


def _construct_voltage_lift(g, deadline):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses a STRUCTURED search (base + small perturbations) with a gcd pre-filter
    and vectorized constraint checking.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]

    n_base = 10
    n_e = len(pet_edges)
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # ---- Precompute unique coefficient vectors of non-backtracking closed
    #      walks of length < g in the Petersen graph. ----
    coeff_set = set()
    max_walk_len = g - 1

    for start_v in range(n_base):
        # stack: (current, prev, length, counts)
        stack = [(start_v, -1, 0, [0] * n_e)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in adj[v]:
                if next_v == prev_v:
                    continue
                e_idx = edge_index[(v, next_v)]
                u_edge, v_edge = pet_edges[e_idx]
                sign = 1 if (v == u_edge and next_v == v_edge) else -1
                new_counts = counts.copy()
                new_counts[e_idx] += sign
                new_l = l + 1
                if next_v == start_v and new_l >= 3:
                    c = tuple(new_counts)
                    # sign-normalize: c and -c are the same constraint
                    f = -1
                    for i in range(n_e):
                        if c[i] != 0:
                            f = i
                            break
                    if f != -1 and c[f] < 0:
                        c = tuple(-x for x in c)
                    coeff_set.add(c)
                stack.append((next_v, v, new_l, new_counts))

    if not coeff_set:
        return None

    coeff_matrix = np.array(list(coeff_set), dtype=np.int64)
    n_constraints = coeff_matrix.shape[0]

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # Candidate moduli. Moore bound for n(3,16) is 512 => m >= 52.
    # Try primes (clean) and a few composites (more voltage freedom).
    primes = []
    cand = 53
    while len(primes) < 40:
        if is_prime(cand):
            primes.append(cand)
        cand += 1

    m_candidates = primes + [52, 54, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 75, 76]
    m_candidates = sorted(set(m_candidates))

    best_n = float('inf')
    best_edges = None

    for m in m_candidates:
        if time.time() > deadline - 15:
            break
        n_candidate = n_base * m
        if n_candidate >= best_n:
            continue

        found = _solve_voltage_structured(coeff_matrix, m, deadline)
        if found is not None:
            edges = _build_lifted_graph(pet_edges, found, m)
            if _is_valid_graph(n_candidate, edges, k=3) and _has_girth_at_least(n_candidate, edges, g):
                return n_candidate, edges

    return None


def _solve_voltage_structured(coeff_matrix, m, deadline):
    """
    Structured search for a voltage vector v in (Z_m)^(n_vars) such that
    coeff_matrix @ v != 0 (mod m) for every row.
    Combines a base assignment with many small perturbations (local polish),
    a gcd pre-filter, and a vectorized cost.
    """
    n_vars = coeff_matrix.shape[1]
    rng = random.Random(12345 + m)

    def cost(v):
        residues = np.mod(coeff_matrix @ v, m)
        return int(np.sum(residues == 0))

    def gcd_all(vals):
        g = m
        for x in vals:
            g = math.gcd(g, x)
            if g == 1:
                break
        return g

    # ---- Base assignment (fixed, deterministic per m) ----
    base = np.array([rng.randint(1, m - 1) for _ in range(n_vars)], dtype=np.int64)
    # Try a few base seeds
    base_seeds = [base]
    for s in range(3):
        b = np.array([rng.randint(1, m - 1) for _ in range(n_vars)], dtype=np.int64)
        base_seeds.append(b)

    # Small perturbations to explore a smooth neighborhood of the base
    deltas = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]

    # Time budget for this m
    m_budget = 12.0 if m < 60 else 7.0
    m_start = time.time()

    for b in base_seeds:
        # check the base itself
        if gcd_all(b.tolist()) == 1:
            if cost(b) == 0:
                return b
        # structured perturbations
        while time.time() - m_start < m_budget:
            if time.time() > deadline - 3:
                return None
            j = rng.randrange(n_vars)
            d = rng.choice(deltas)
            nv = b[j] + d
            if nv <= 0 or nv >= m:
                continue
            v = b.copy()
            v[j] = nv
            if gcd_all(v.tolist()) != 1:
                continue
            if cost(v) == 0:
                return v
    return None


def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            sign = 1 if (u == u_edge and v == v_edge) else -1
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    return list(edges)


def _is_valid_graph(n, edges, k):
    if len(edges) != n * k // 2:
        return False
    degree = [0] * n
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        degree[u] += 1
        degree[v] += 1
    for d in degree:
        if d != k:
            return False
    return True


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


def _has_girth_at_least(n, edges, g):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    max_len = g - 1
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            u, prev, l = queue.popleft()
            if u == start and l > 0:
                return False
            if l >= max_len:
                continue
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    return True
