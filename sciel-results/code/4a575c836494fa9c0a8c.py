import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_long_cycle(adj, n, max_starts=30, max_restarts=8, rng=None):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy with improved scoring.
    Returns the cycle as a list of vertices or None.
    """
    if rng is None:
        rng = random.Random(42)

    best_cycle = None
    best_len = 0

    # Try starting from each vertex with some randomization
    num_starts = min(n, max_starts)
    if n <= max_starts:
        starts = list(range(n))
    else:
        # Prioritize high-degree vertices
        degrees = [len(adj[v]) for v in range(n)]
        starts = rng.sample(range(n), num_starts)

    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]

                if not unvisited:
                    # Check if we can close the cycle
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Heuristic: prefer neighbors with more unvisited connections
                    # Also consider closing early if path is already long
                    if len(path) >= 4 and start in adj[current] and rng.random() < 0.3:
                        # Close the cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    # Score each unvisited neighbor
                    scored = []
                    for v in unvisited:
                        # Count unvisited neighbors of v
                        unvisited_neighbors = sum(1 for w in adj[v] if w not in visited)
                        score = unvisited_neighbors
                        # Slight preference for closing to start if possible
                        if v == start:
                            score += 10
                        scored.append((score, v))

                    # Pick the best or randomly from top few
                    scored.sort(reverse=True)
                    top_k = min(3, len(scored))
                    if rng.random() < 0.7:
                        best_next = scored[0][1]
                    else:
                        best_next = rng.choice([s[1] for s in scored[:top_k]])

                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next

    return best_cycle


def _decompose_greedy(edges, n, deadline, rng=None):
    """
    Greedy decomposition: repeatedly find a cycle and remove it.
    Uses a randomized greedy path extension to find cycles.
    """
    if rng is None:
        rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        # Try to find a long cycle
        best_cycle = _find_long_cycle(adj, n, max_starts=min(n, 25), max_restarts=6, rng=rng)

        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            # No cycle found, remove a single edge
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    # Add any remaining edges as single edges
    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _decompose_flow(edges, n, deadline):
    """
    Decompose using flow-based approach:
    1. Create directed graph with both directions for each undirected edge
    2. Find maximum flow decomposition into cycles and paths
    3. Pair path endpoints to close them into cycles
    """
    from scipy.optimize import linear_sum_assignment

    # Build directed graph
    # For each undirected edge (u,v), create two directed edges: u->v and v->u
    # We want to decompose into directed cycles and paths, then close paths

    # First, let's try a simpler approach: find a large cycle using the greedy method
    # then handle remaining edges
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining_edges = set(tuple(sorted(e)) for e in edges)

    # Iteratively find cycles
    while remaining_edges:
        if time.time() > deadline:
            break

        # Find a cycle in the remaining graph
        best_cycle = None
        best_len = 0

        # Try from each vertex
        for start in range(n):
            if not adj[start]:
                continue

            # DFS to find a cycle back to start
            stack = [(start, [start])]
            visited_in_dfs = {start}

            while stack:
                node, path = stack.pop()

                for neighbor in list(adj[node]):
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break
                    elif neighbor not in visited_in_dfs:
                        visited_in_dfs.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))

        if best_cycle:
            # Remove the cycle edges
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                edge = tuple(sorted((a, b)))
                remaining_edges.discard(edge)
                adj[a].discard(b)
                adj[b].discard(a)
            pieces.append(best_cycle)
        else:
            # No cycle, take a single edge
            if remaining_edges:
                u, v = next(iter(remaining_edges))
                remaining_edges.discard((u, v))
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])
            else:
                break

    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try multiple random strategies
    base_seed = 12345
    attempts = 0
    max_attempts = 150

    while time.time() < deadline - 1.0:
        seed = base_seed + attempts
        rng = random.Random(seed)
        pieces = _decompose_greedy(edges, n, deadline - 0.5, rng)
        cnt = len(pieces)

        if cnt < best_count:
            best_count = cnt
            best = pieces

        attempts += 1

        if attempts >= max_attempts:
            break

    # Also try the flow-based approach if time permits
    if time.time() < deadline - 5.0:
        flow_pieces = _decompose_flow(edges, n, deadline - 2.0)
        if len(flow_pieces) < best_count:
            best_count = len(flow_pieces)
            best = flow_pieces

    return best if best else [[u, v] for u, v in edges]


def construct():
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
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
