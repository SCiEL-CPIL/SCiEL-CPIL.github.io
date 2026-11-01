import time
import random
import sys
from collections import deque

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Exact girth check via non-backtracking BFS from every vertex.
    Returns True iff no cycle shorter than min_girth exists."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2
    if max_depth < 1:
        return True
    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])
        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _cube_lift(k, g, deadline):
    """Voltage lift of the 3-cube (Q3) over Z_m.
    Q3: 8 vertices, 12 edges, 3-regular, girth 4.
    Lift has n = 8*m vertices.
    """
    if k != 3:
        return None

    # 3-cube edges
    base_edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # x-dimension
        (0, 2), (1, 3), (4, 6), (5, 7),  # y-dimension
        (0, 4), (1, 5), (2, 6), (3, 7),  # z-dimension
    ]
    n_base = 8
    n_edges = 12

    # Find all 4-cycles in Q3 (there are 6: one per face)
    adj_base = [[] for _ in range(n_base)]
    for idx, (u, v) in enumerate(base_edges):
        adj_base[u].append((v, idx))
        adj_base[v].append((u, idx))

    four_cycles = set()
    for start in range(n_base):
        stack = [(start, start, frozenset(), [start])]
        while stack:
            v, root, used_edges, path = stack.pop()
            if len(path) == 4:
                for nxt, eidx in adj_base[v]:
                    if nxt == root:
                        four_cycles.add(frozenset(used_edges | {eidx}))
            elif len(path) < 4:
                for nxt, eidx in adj_base[v]:
                    if nxt == root and len(path) >= 3:
                        continue
                    if nxt not in path and eidx not in used_edges:
                        stack.append((nxt, root, used_edges | {eidx}, path + [nxt]))

    cyc_edge_lists = [list(c) for c in four_cycles]

    def _cycle_sum_ok(voltages, m):
        s4 = [sum(voltages[i] for i in c) % m for c in cyc_edge_lists]
        for s in s4:
            if s == 0:
                return False
        # 8-cycles: sums of two 4-cycles
        for i in range(len(s4)):
            for j in range(i, len(s4)):
                if (s4[i] + s4[j]) % m == 0:
                    return False
        # 12-cycles: sums of three 4-cycles
        L = len(s4)
        for i in range(L):
            for j in range(L):
                for l in range(L):
                    if (s4[i] + s4[j] + s4[l]) % m == 0:
                        return False
        return True

    def _build_lift(m, voltages):
        lift_edges = []
        for idx, (u, v) in enumerate(base_edges):
            vol = voltages[idx]
            for j in range(m):
                a = u * m + j
                b = v * m + ((j + vol) % m)
                lift_edges.append((a, b))
        return lift_edges

    # Moore bound for k=3, g=16 is 510; target n = 8*m
    moore = 2 + 3 * sum(2 ** t for t in range((g - 1) // 2))
    min_m = max(64, (moore // 8) + 1)
    max_m = 130

    for m in range(min_m, max_m + 1):
        if time.time() > deadline:
            break
        for attempt in range(6000):
            if time.time() > deadline:
                break
            voltages = [random.randrange(m) for _ in range(n_edges)]
            if not _cycle_sum_ok(voltages, m):
                continue
            lift_edges = _build_lift(m, voltages)
            n = 8 * m
            if _check_girth_fast(n, lift_edges, g):
                return n, lift_edges
    return None


def _petersen_lift(k, g, deadline):
    """Voltage lift of the Petersen graph over Z_m.
    Petersen: 10 vertices, 15 edges, 3-regular, girth 5.
    Lift has n = 10*m vertices. Find m and voltages so the lift has girth >= g.
    """
    if k != 3:
        return None

    # Petersen graph edges (outer pentagon + inner star + spokes)
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),          # outer 5-cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),          # inner 5-point star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),          # spokes
    ]
    n_base = 10
    n_edges = 15
    n_base_cycle = 5

    # Precompute the 12 five-cycles of the Petersen graph as edge-index sets.
    adj_base = [[] for _ in range(n_base)]
    for idx, (u, v) in enumerate(base_edges):
        adj_base[u].append((v, idx))
        adj_base[v].append((u, idx))

    # Enumerate all 5-cycles (edge sets) in the Petersen graph.
    five_cycles = set()
    for start in range(n_base):
        stack = [(start, start, frozenset(), [start])]
        while stack:
            v, root, used_edges, path = stack.pop()
            if len(path) == n_base_cycle:
                for nxt, eidx in adj_base[v]:
                    if nxt == root:
                        five_cycles.add(frozenset(used_edges | {eidx}))
            elif len(path) < n_base_cycle:
                for nxt, eidx in adj_base[v]:
                    if nxt == root and len(path) >= n_base_cycle - 1:
                        continue
                    if nxt not in path and eidx not in used_edges:
                        stack.append((nxt, root, used_edges | {eidx}, path + [nxt]))

    cyc_edge_lists = [list(c) for c in five_cycles]

    def _cycle_sum_ok(voltages, m):
        s5 = [sum(voltages[i] for i in c) % m for c in cyc_edge_lists]
        for s in s5:
            if s == 0:
                return False
        for i in range(len(s5)):
            for j in range(i, len(s5)):
                if (s5[i] + s5[j]) % m == 0:
                    return False
        L = len(s5)
        for i in range(L):
            for j in range(L):
                for l in range(L):
                    if (s5[i] + s5[j] + s5[l]) % m == 0:
                        return False
        return True

    def _build_lift(m, voltages):
        lift_edges = []
        for idx, (u, v) in enumerate(base_edges):
            vol = voltages[idx]
            for j in range(m):
                a = u * m + j
                b = v * m + ((j + vol) % m)
                lift_edges.append((a, b))
        return lift_edges

    moore = 2 + 3 * sum(2 ** t for t in range((g - 1) // 2))
    min_m = max(52, (moore // 10) + 1)
    max_m = 100

    for m in range(min_m, max_m + 1):
        if time.time() > deadline:
            break
        for attempt in range(4000):
            if time.time() > deadline:
                break
            voltages = [random.randrange(m) for _ in range(n_edges)]
            if not _cycle_sum_ok(voltages, m):
                continue
            lift_edges = _build_lift(m, voltages)
            n = 10 * m
            if _check_girth_fast(n, lift_edges, g):
                return n, lift_edges
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for 3-regular graphs."""
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3
    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def construct(k, g):
    start_time = time.time()
    deadline = 135.0  # safety margin under 150s

    if k != 3:
        return _luw_construct(k, g)

    # Try cube lift first (different algebraic structure, 8*m vertices)
    result = _cube_lift(k, g, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Try Petersen lift
    result = _petersen_lift(k, g, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
