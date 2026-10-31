import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) as fallback."""
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


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_nb(n, edges, min_girth):
    """
    Fast girth certification using non-backtracking walks from every vertex.
    A cycle of length < min_girth exists iff there is a non-backtracking closed
    walk of length < min_girth. We do BFS on the non-backtracking state space
    (vertex, parent) up to depth (min_girth-1)//2, checking for collisions.
    """
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        # State: (current_vertex, parent_vertex)
        # visited[state] = depth
        visited = {}
        # Initial states: for each neighbor of start, state (neighbor, start) at depth 1
        for nb in adj[start]:
            visited[(nb, start)] = 1

        # BFS
        frontier = [(nb, start) for nb in adj[start]]
        depth = 1
        while frontier and depth < max_depth:
            next_frontier = []
            for (u, parent) in frontier:
                for w in adj[u]:
                    if w == parent:
                        continue
                    state = (w, u)
                    if state in visited:
                        # Found a cycle of length depth + visited[state]
                        cycle_len = depth + visited[state]
                        if cycle_len < min_girth:
                            return False
                    else:
                        visited[state] = depth + 1
                        next_frontier.append(state)
            frontier = next_frontier
            depth += 1
    return True


def _get_petersen_cycles_up_to(max_len):
    """
    Pre-compute all cycles in the Petersen graph up to max_len.
    Returns a list of tuples of edge indices (canonicalized).
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n = 10
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    seen = set()
    cycles = []

    def dfs(start, curr, visited, edge_path, length):
        if length > max_len:
            return
        for next_v, edge_idx in adj[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)

    for start in range(n):
        visited = {start}
        dfs(start, start, visited, [], 0)

    return cycles


def _quadratic_residues(m):
    """Return the set of quadratic residues mod m."""
    if m <= 1:
        return {0}
    residues = set()
    for x in range(m):
        residues.add((x * x) % m)
    return residues


def _find_voltages_deterministic(m, cycles, deadline):
    """
    Try to find voltages for Petersen edges in Z_m such that no forbidden
    cycle sums to 0 mod m. Use quadratic residues as initial assignment,
    then local search.
    """
    n_edges = 15
    qr = _quadratic_residues(m)
    qr_list = sorted(qr)

    # Initialize with quadratic residues (cyclically assigned)
    voltages = [qr_list[i % len(qr_list)] for i in range(n_edges)]

    def count_bad():
        bad = 0
        for cycle in cycles:
            s = sum(voltages[i] for i in cycle) % m
            if s == 0:
                bad += 1
        return bad

    current_bad = count_bad()
    if current_bad == 0:
        return voltages

    # Local search: try to fix bad cycles
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        if current_bad == 0:
            return voltages

        # Find a bad cycle
        bad_cycle = None
        for cycle in cycles:
            if sum(voltages[i] for i in cycle) % m == 0:
                bad_cycle = cycle
                break

        if bad_cycle is None:
            return voltages

        # Try changing each edge in the bad cycle
        improved = False
        for edge_idx in bad_cycle:
            old_v = voltages[edge_idx]
            # Try a few random values
            for _ in range(5):
                new_v = random.randint(0, m - 1)
                if new_v == old_v:
                    continue
                voltages[edge_idx] = new_v
                new_bad = count_bad()
                if new_bad < current_bad:
                    current_bad = new_bad
                    improved = True
                    break
                else:
                    voltages[edge_idx] = old_v

        if not improved:
            # Random restart
            voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
            current_bad = count_bad()

    return voltages if current_bad == 0 else None


def _find_voltages_random(m, cycles, deadline, seed=0):
    """Random search for voltages."""
    rng = random.Random(seed)
    n_edges = 15

    def count_bad(voltages):
        bad = 0
        for cycle in cycles:
            s = sum(voltages[i] for i in cycle) % m
            if s == 0:
                bad += 1
        return bad

    best_voltages = [rng.randint(0, m - 1) for _ in range(n_edges)]
    best_bad = count_bad(best_voltages)
    if best_bad == 0:
        return best_voltages

    for it in range(20000):
        if time.time() > deadline:
            return None
        if best_bad == 0:
            return best_voltages

        # Pick a random edge and change its voltage
        idx = rng.randint(0, n_edges - 1)
        old_v = best_voltages[idx]
        new_v = rng.randint(0, m - 1)
        if new_v == old_v:
            continue
        best_voltages[idx] = new_v
        new_bad = count_bad(best_voltages)
        if new_bad <= best_bad:
            best_bad = new_bad
        else:
            best_voltages[idx] = old_v

    return best_voltages if best_bad == 0 else None


def _petersen_lift_with_voltages(m, voltages):
    """Construct the lift using specific voltages."""
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n = 10 * m
    edges = []
    for idx, (u, v) in enumerate(petersen_edges):
        vol = voltages[idx]
        for j in range(m):
            a = u * m + j
            b = v * m + ((j + vol) % m)
            edges.append((a, b))

    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    if k != 3:
        return _luw_construct(k, g)

    # Pre-compute cycles in Petersen graph up to g-1
    cycles_to_forbid = _get_petersen_cycles_up_to(g - 1)

    # Moore bound for (3, g): roughly 2 * 3^(g/2 - 1) for even g
    # For g=16: 2 * 3^7 = 4374, but actual Moore bound is lower.
    # Best known lower bound is 512. Let's start m from 52 (n=520) upward.
    # n = 10 * m, so m = n/10.
    # Start from m=52 and increment by 1.

    best_result = None
    best_n = float('inf')

    # Try deterministic quadratic residue approach first for small m
    for m in range(52, 100):
        if time.time() > deadline - 20:
            break

        voltages = _find_voltages_deterministic(m, cycles_to_forbid, deadline - 20)
        if voltages is not None:
            n, edges = _petersen_lift_with_voltages(m, voltages)
            if _check_girth_nb(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                # Found a valid one, try to find smaller
                # Since we're scanning upward, the first valid one is the smallest for this approach
                return n, edges

    # Try random voltage search for each m
    for m in range(52, 200):
        if time.time() > deadline - 10:
            break

        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            voltages = _find_voltages_random(m, cycles_to_forbid, deadline - 10, seed=attempt * 1000 + m)
            if voltages is not None:
                n, edges = _petersen_lift_with_voltages(m, voltages)
                if _check_girth_nb(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    return n, edges

    # If we found something, return it
    if best_result is not None:
        return best_result

    # Fallback: LUW construction
    return _luw_construct(k, g)
