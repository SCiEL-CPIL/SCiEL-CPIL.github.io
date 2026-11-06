"""Decompose three fixed graphs into cycles and edges, minimizing piece count.

Strategy:
1. Find all simple cycles up to a reasonable length bound (e.g., 12) via DFS.
2. Formulate an Integer Linear Program (ILP) using scipy.optimize.milp:
   - Variables: x_c for each cycle c (1 if used, 0 otherwise)
   - Objective: minimize (number of cycles used) + (number of edges not covered by cycles)
     Since each edge not in a cycle becomes a singleton piece, and each cycle is one piece,
     total pieces = sum(x_c) + (total_edges - sum(length_c * x_c))
     = total_edges + sum(x_c * (1 - length_c))
     So we minimize sum(x_c * (1 - length_c)).
   - Constraints: each edge is covered by at most one cycle (edge-disjointness).
3. Solve the ILP. If it fails or takes too long, fall back to a greedy heuristic.
4. Remaining edges not covered by selected cycles become singleton pieces.
"""

import time
import random
from collections import defaultdict
from itertools import combinations

_DEADLINE_SECONDS = 140.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycles(adj, n, max_len=12, time_limit=5.0):
    """
    Find all simple cycles of length up to max_len in the graph.
    Returns a list of cycles, where each cycle is a tuple of vertices.
    We normalize cycles to avoid duplicates (start with smallest vertex, canonical rotation).
    """
    start_time = time.time()
    cycles = []
    seen = set()

    def canonical_cycle(cycle):
        """Return a canonical representation of the cycle to avoid duplicates."""
        if not cycle:
            return None
        # Find the minimum vertex
        min_idx = cycle.index(min(cycle))
        # Rotate so min vertex is first
        rotated = cycle[min_idx:] + cycle[:min_idx]
        # Check reverse
        reversed_rotated = [rotated[0]] + rotated[1:][::-1]
        # Choose the lexicographically smaller
        if tuple(rotated) <= tuple(reversed_rotated):
            return tuple(rotated)
        else:
            return tuple(reversed_rotated)

    def dfs(start, current, visited, path):
        nonlocal cycles, seen
        if time.time() - start_time > time_limit:
            return

        if len(path) >= 3 and current in adj.get(start, set()):
            # Found a cycle
            cycle = path[:]
            if len(cycle) <= max_len:
                canon = canonical_cycle(cycle)
                if canon and canon not in seen:
                    seen.add(canon)
                    cycles.append(canon)

        for nb in sorted(adj.get(current, set())):
            if nb == start:
                if len(path) >= 3:
                    cycle = path[:]
                    if len(cycle) <= max_len:
                        canon = canonical_cycle(cycle)
                        if canon and canon not in seen:
                            seen.add(canon)
                            cycles.append(canon)
                continue
            if nb not in visited and nb > start:
                # Only visit vertices greater than start to avoid duplicates
                visited.add(nb)
                path.append(nb)
                dfs(start, nb, visited, path)
                path.pop()
                visited.remove(nb)

    for start in range(n):
        if time.time() - start_time > time_limit:
            break
        if len(adj.get(start, set())) < 2:
            continue
        visited = {start}
        path = [start]
        dfs(start, start, visited, path)

    return cycles


def _decompose_with_ilp(edges, n, time_limit=10.0):
    """
    Decompose graph using ILP to select edge-disjoint cycles maximizing coverage.
    """
    if not edges:
        return []

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    total_edges = len(edge_set)

    if total_edges == 0:
        return []

    # Build adjacency
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    # Find cycles
    start_time = time.time()
    cycles = _find_cycles(adj, n, max_len=10, time_limit=min(5.0, time_limit * 0.5))

    if not cycles:
        # No cycles found, return all edges as singletons
        return [[u, v] for u, v in edges]

    # If time is running out, fall back to greedy
    if time.time() - start_time > time_limit * 0.6:
        return _greedy_decompose(edge_set, n, time_limit * 0.4)

    # Set up ILP
    # Variables: x_i for each cycle i
    m = len(cycles)
    if m == 0:
        return [[u, v] for u, v in edges]

    # Objective: minimize sum(x_i * (1 - len(cycle_i)))
    # Since total_pieces = total_edges + sum(x_i * (1 - len(cycle_i)))
    c = [1 - len(cycle) for cycle in cycles]

    # Constraints: for each edge, sum of x_i over cycles containing that edge <= 1
    # Build edge-to-cycle incidence
    edge_to_cycles = defaultdict(list)
    for i, cycle in enumerate(cycles):
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            edge_to_cycles[e].append(i)

    # Only consider edges that appear in at least one cycle
    relevant_edges = [e for e in edge_set if e in edge_to_cycles]

    A_rows = []
    b_vals = []
    for e in relevant_edges:
        row = [0] * m
        for i in edge_to_cycles[e]:
            row[i] = 1
        A_rows.append(row)
        b_vals.append(1)

    if not A_rows:
        # No edges covered by any cycle
        return [[u, v] for u, v in edges]

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Variables are binary
        integrality = [1] * m
        bounds = Bounds(lb=0, ub=1)

        # Constraints: A @ x <= b
        if A_rows:
            A = A_rows
            constraints = LinearConstraint(A, ub=b_vals)
        else:
            constraints = []

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints,
                   options={'time_limit': max(1.0, time_limit * 0.4)})

        if res.status != 0 and res.status != 1:  # 0 = optimal, 1 = feasible
            # Fallback to greedy
            return _greedy_decompose(edge_set, n, time_limit * 0.3)

        # Extract selected cycles
        selected_cycles = []
        used_edges = set()
        for i in range(m):
            if res.x is not None and res.x[i] > 0.5:
                cycle = cycles[i]
                # Check if all edges are still available
                cycle_edges = []
                valid = True
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    cycle_edges.append(e)
                if valid:
                    selected_cycles.append(cycle)
                    used_edges.update(cycle_edges)

        # Remaining edges
        remaining_edges = [e for e in edge_set if e not in used_edges]

        pieces = []
        for cycle in selected_cycles:
            pieces.append(list(cycle))
        for u, v in remaining_edges:
            pieces.append([u, v])

        return pieces

    except Exception:
        # Fallback to greedy
        return _greedy_decompose(edge_set, n, time_limit * 0.3)


def _greedy_decompose(edge_set, n, time_limit=5.0):
    """
    Greedy decomposition: repeatedly find a long cycle (via DFS) and remove it.
    """
    if not edge_set:
        return []

    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    deg = [len(adj[v]) for v in range(n)]
    pieces = []
    start_time = time.time()
    rng = random.Random(42)

    while True:
        if time.time() - start_time > time_limit:
            break

        # Check remaining edges
        remaining = sum(len(adj[u]) for u in range(n)) // 2
        if remaining == 0:
            break

        # Find a cycle
        cycle = None
        candidates = [v for v in range(n) if deg[v] >= 2]
        if not candidates:
            # No cycles, take an edge
            if adj:
                u = next(iter(adj))
                if adj[u]:
                    v = next(iter(adj[u]))
                    pieces.append([u, v])
                    adj[u].discard(v)
                    adj[v].discard(u)
                    deg[u] -= 1
                    deg[v] -= 1
            continue

        # Try a few random DFS runs to find a cycle
        best_cycle = None
        best_len = 0
        step_start = time.time()
        max_search_time = 0.05

        for _ in range(5):
            if time.time() - step_start > max_search_time:
                break
            start = rng.choice(candidates)
            # DFS to find a cycle
            stack = [(start, [start], {start})]
            while stack:
                if time.time() - step_start > max_search_time:
                    break
                node, path, visited = stack.pop()
                neighbors = list(adj[node])
                neighbors.sort(key=lambda x: (x in visited, -deg[x]))
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    elif nb not in visited:
                        new_visited = visited | {nb}
                        stack.append((nb, path + [nb], new_visited))

        if best_cycle:
            pieces.append(best_cycle)
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No cycle found, take an edge
            if adj:
                u = next(iter(adj))
                if adj[u]:
                    v = next(iter(adj[u]))
                    pieces.append([u, v])
                    adj[u].discard(v)
                    adj[v].discard(u)
                    deg[u] -= 1
                    deg[v] -= 1

    # Add remaining edges
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    return pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 15:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        # Allocate time for this graph
        graph_time = remaining * 0.9

        # Try ILP first
        pieces = _decompose_with_ilp(edges, n, graph_time)

        results.append(pieces)

    return results
