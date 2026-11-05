import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycles_dynamic(adj, n, edge_set, rng, time_limit=3.0):
    """
    Dynamically find cycles in the current graph.
    Uses multiple strategies to find long cycles.
    """
    start_time = time.time()
    cycles = []
    seen = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_idx = cycle.index(min(cycle))
        c1 = cycle[min_idx:] + cycle[:min_idx]
        c2 = list(reversed(c1))
        return tuple(c1) if c1 < c2 else tuple(c2)

    def _add_cycle(cycle):
        if len(cycle) < 3:
            return
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return
        key = _canonical(cycle)
        if key and key not in seen:
            seen.add(key)
            cycles.append(list(cycle))

    # Strategy 1: Find cycles via DFS from each vertex
    for start in range(n):
        if time.time() - start_time > time_limit:
            break
        if not adj[start]:
            continue
        for _ in range(2):
            visited = {start}
            path = [start]
            current = start
            max_depth = min(n, 20)
            depth = 0
            while depth < max_depth:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                if start in adj[current] and len(path) >= 3:
                    _add_cycle(path[:])
                if not unvisited:
                    break
                # Prefer neighbors that lead to longer cycles
                scored = sorted(unvisited, key=lambda v: len(adj[v]), reverse=True)
                if rng.random() < 0.4:
                    next_v = rng.choice(unvisited)
                else:
                    top_k = min(3, len(scored))
                    next_v = rng.choice(scored[:top_k])
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                depth += 1

    # Strategy 2: Random walks
    for _ in range(150):
        if time.time() - start_time > time_limit:
            break
        start = rng.choice(range(n))
        if not adj[start]:
            continue
        path = [start]
        visited = {start}
        current = start
        for _ in range(n):
            neighbors = list(adj[current])
            if start in neighbors and len(path) >= 3:
                _add_cycle(path[:])
                break
            unvisited = [v for v in neighbors if v not in visited]
            if not unvisited:
                break
            next_v = rng.choice(unvisited)
            path.append(next_v)
            visited.add(next_v)
            current = next_v

    # Strategy 3: Triangles
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    _add_cycle([u, v, w])

    return cycles


def _solve_one(edges, n, deadline):
    """
    Solve one instance: decompose edges into minimum number of pieces.
    Uses dynamic cycle extraction with local search.
    """
    if not edges:
        return []

    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    if m < 3:
        return [list(e) for e in edges]

    # Check if graph has any cycles
    # If it's a forest, all edges are single pieces
    if m < n:
        # Could still have one cycle if m == n, but for m < n it's a forest
        return [list(e) for e in edges]

    best_pieces = None
    best_count = float('inf')

    max_restarts = 40
    for attempt in range(max_restarts):
        if time.time() > deadline - 0.5:
            break

        rng_attempt = random.Random(attempt * 137 + 42)
        remaining = set(edge_set)
        current_adj = [set() for _ in range(n)]
        for u, v in remaining:
            current_adj[u].add(v)
            current_adj[v].add(u)

        selected_cycles = []
        used_edges = set()

        # Dynamically extract cycles
        while time.time() < deadline - 0.5:
            # Find cycles in the current remaining graph
            cycles = _find_cycles_dynamic(current_adj, n, remaining, rng_attempt, time_limit=1.5)

            if not cycles:
                break

            # Sort cycles by length (prefer longer cycles)
            cycles.sort(key=len, reverse=True)

            # Try to add the best cycle
            added = False
            for cycle in cycles:
                cycle_edges = set()
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add(tuple(sorted((a, b))))

                if cycle_edges.issubset(remaining):
                    # Check edge-disjointness
                    if not (cycle_edges & used_edges):
                        selected_cycles.append(cycle)
                        used_edges |= cycle_edges
                        remaining -= cycle_edges
                        # Update adjacency
                        for u, v in cycle_edges:
                            current_adj[u].discard(v)
                            current_adj[v].discard(u)
                        added = True
                        break

            if not added:
                break

        # Local search: try to improve by swapping cycles
        for ls_iter in range(50):
            if time.time() > deadline - 0.5:
                break
            improved = False

            # Try to remove a cycle and add a better one
            for i, cycle in enumerate(selected_cycles):
                cycle_edges = set()
                for j in range(len(cycle)):
                    a = cycle[j]
                    b = cycle[(j + 1) % len(cycle)]
                    cycle_edges.add(tuple(sorted((a, b))))

                # Temporarily remove this cycle
                temp_remaining = remaining | cycle_edges
                temp_used = used_edges - cycle_edges

                # Find a better cycle in temp_remaining
                temp_adj = [set() for _ in range(n)]
                for u, v in temp_remaining:
                    temp_adj[u].add(v)
                    temp_adj[v].add(u)

                new_cycles = _find_cycles_dynamic(temp_adj, n, temp_remaining, rng_attempt, time_limit=0.5)
                new_cycles.sort(key=len, reverse=True)

                for new_cycle in new_cycles:
                    new_cycle_edges = set()
                    for j in range(len(new_cycle)):
                        a = new_cycle[j]
                        b = new_cycle[(j + 1) % len(new_cycle)]
                        new_cycle_edges.add(tuple(sorted((a, b))))

                    if new_cycle_edges.issubset(temp_remaining) and not (new_cycle_edges & temp_used):
                        # Check if this improves the solution
                        old_len = len(cycle)
                        new_len = len(new_cycle)
                        if new_len > old_len:
                            # Replace
                            selected_cycles[i] = new_cycle
                            used_edges = temp_used | new_cycle_edges
                            remaining = temp_remaining - new_cycle_edges
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                break

        # Compute pieces
        pieces = [list(c) for c in selected_cycles]
        for e in remaining:
            pieces.append(list(e))

        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [list(e) for e in edges]

    return best_pieces


def construct():
    start_time = time.time()
    total_budget = 140.0  # Safety margin for 150s limit
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
