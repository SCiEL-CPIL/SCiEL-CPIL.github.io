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


def _find_cycles_pool(adj, n, edge_set, num_cycles=500, rng=None, time_limit=5.0):
    """
    Generate a pool of candidate simple cycles using multiple strategies.
    Returns a list of cycles, each a list of vertices (length >= 3).
    """
    if rng is None:
        rng = random.Random(12345)
    start_time = time.time()
    cycles = []
    seen = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        # Canonical form: rotate so smallest vertex is first, and pick direction
        min_idx = cycle.index(min(cycle))
        c1 = cycle[min_idx:] + cycle[:min_idx]
        c2 = list(reversed(c1))
        return tuple(c1) if c1 < c2 else tuple(c2)

    def _add_cycle(cycle):
        if len(cycle) < 3:
            return
        # Verify all edges exist
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

    # Strategy 1: DFS-based cycle finding from each vertex
    for start in range(n):
        if time.time() - start_time > time_limit:
            break
        if not adj[start]:
            continue
        # Randomized DFS
        for _ in range(3):
            visited = {start}
            path = [start]
            current = start
            max_depth = n
            depth = 0
            while depth < max_depth:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                if start in adj[current] and len(path) >= 3:
                    _add_cycle(path[:])
                if not unvisited:
                    break
                # Prefer low-degree neighbors for longer paths
                scored = sorted(unvisited, key=lambda v: len(adj[v]))
                if rng.random() < 0.3:
                    next_v = rng.choice(unvisited)
                else:
                    top_k = min(3, len(scored))
                    next_v = rng.choice(scored[:top_k])
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                depth += 1

    # Strategy 2: Random walks that close into cycles
    for _ in range(200):
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

    # Strategy 3: Find cycles by looking at pairs of common neighbors
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    # Triangle u-v-w-u
                    _add_cycle([u, v, w])

    return cycles


def _cycle_edges(cycle):
    """Return the set of edges in a cycle."""
    edges = set()
    m = len(cycle)
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        edges.add(tuple(sorted((a, b))))
    return edges


def _solve_one(edges, n, deadline):
    """
    Solve one instance: decompose edges into minimum number of pieces.
    Uses cycle pool + greedy/local search to select edge-disjoint cycles.
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

    # Check if graph has any cycles (if it's a forest, all edges are single pieces)
    # Quick check: if m < n, it's a forest (or has at most one cycle per component)
    # Actually, let's just proceed with the algorithm

    # Generate cycle pool
    rng = random.Random(777)
    cycles_pool = _find_cycles_pool(adj, n, edge_set, num_cycles=800, rng=rng, time_limit=8.0)

    if not cycles_pool:
        return [list(e) for e in edges]

    # Precompute edge sets for each cycle
    pool_edge_sets = [_cycle_edges(c) for c in cycles_pool]
    pool_lengths = [len(c) for c in cycles_pool]
    pool_cycles = cycles_pool

    # Objective: minimize pieces = m - 2*sum(|c_i|) + k where k = number of cycles
    # Equivalent to maximize: 2*sum(|c_i|) - k
    # For a set of edge-disjoint cycles S: value = sum_{i in S} (2*|c_i| - 1)

    # Greedy: repeatedly pick the cycle with best (2*len - 1) / |edges| ratio that doesn't conflict
    def _greedy_select(pool_cycles, pool_edge_sets, pool_lengths, remaining_edges):
        selected = []
        used_edges = set()
        available = list(range(len(pool_cycles)))

        while available:
            # Score each available cycle: (2*len - 1)
            best_idx = -1
            best_score = -1
            for idx in available:
                if not pool_edge_sets[idx].issubset(remaining_edges):
                    continue
                score = 2 * pool_lengths[idx] - 1
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            used_edges |= pool_edge_sets[best_idx]
            available.remove(best_idx)

        return selected, used_edges

    # Local search: try to swap a selected cycle for a non-selected one if it improves
    def _local_search(selected, used_edges, remaining_edges, max_iters=200):
        selected_set = set(selected)
        for _ in range(max_iters):
            if time.time() > deadline - 0.5:
                break
            improved = False
            # Try adding a cycle
            for idx in range(len(pool_cycles)):
                if idx in selected_set:
                    continue
                if not pool_edge_sets[idx].issubset(remaining_edges):
                    continue
                # Add it
                new_selected = selected + [idx]
                new_used = used_edges | pool_edge_sets[idx]
                new_remaining = remaining_edges - pool_edge_sets[idx]
                # Try to remove a cycle to improve
                for j in selected:
                    # Remove j, add idx
                    candidate_selected = [x for x in selected if x != j] + [idx]
                    candidate_used = (used_edges - pool_edge_sets[j]) | pool_edge_sets[idx]
                    if not candidate_used.issubset(edge_set):
                        continue
                    # Check edge-disjointness
                    cand_remaining = edge_set - candidate_used
                    # All good if candidate_used is valid
                    # Compute objective
                    old_val = sum(2 * pool_lengths[i] - 1 for i in selected)
                    new_val = sum(2 * pool_lengths[i] - 1 for i in candidate_selected)
                    if new_val > old_val:
                        selected = candidate_selected
                        used_edges = candidate_used
                        remaining_edges = cand_remaining
                        selected_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
            # Try removing a cycle
            if not improved and selected:
                for j in selected:
                    candidate_selected = [x for x in selected if x != j]
                    candidate_used = used_edges - pool_edge_sets[j]
                    old_val = sum(2 * pool_lengths[i] - 1 for i in selected)
                    new_val = sum(2 * pool_lengths[i] - 1 for i in candidate_selected)
                    # Removing always decreases value, so only remove if it enables better additions
                    # Skip for now
                    pass
            if not improved:
                break
        return selected, used_edges, edge_set - used_edges

    best_pieces = None
    best_count = float('inf')

    # Multiple randomizations
    max_restarts = 30
    for attempt in range(max_restarts):
        if time.time() > deadline - 0.3:
            break

        # Shuffle pool order for diversity
        rng_attempt = random.Random(attempt * 100 + 42)
        shuffled_indices = list(range(len(cycles_pool)))
        rng_attempt.shuffle(shuffled_indices)

        # Greedy with different starting orders
        # Try a few different greedy strategies
        for strategy in range(3):
            if time.time() > deadline - 0.3:
                break

            if strategy == 0:
                # Standard greedy: best score first
                order = sorted(shuffled_indices, key=lambda i: 2 * pool_lengths[i] - 1, reverse=True)
            elif strategy == 1:
                # Greedy: prefer longer cycles
                order = sorted(shuffled_indices, key=lambda i: pool_lengths[i], reverse=True)
            else:
                # Random order
                order = shuffled_indices[:]

            remaining = set(edge_set)
            selected = []
            used_edges = set()

            for idx in order:
                if time.time() > deadline - 0.3:
                    break
                if not pool_edge_sets[idx].issubset(remaining):
                    continue
                # Accept if it improves objective (always does since 2*len-1 > 0 for len>=3)
                selected.append(idx)
                used_edges |= pool_edge_sets[idx]
                remaining -= pool_edge_sets[idx]

            # Local search
            selected, used_edges, remaining = _local_search(selected, used_edges, remaining, max_iters=100)

            # Compute pieces
            pieces = [list(cycles_pool[i]) for i in selected]
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
