"""Rewritten program: decompose each instance graph into edge-disjoint cycles and edges,
minimizing total number of pieces. Uses a greedy cycle-extraction heuristic with
multiple strategies and a time guard."""
import time
from collections import defaultdict, deque
from random import Random

START_TIME = time.time()
TIME_LIMIT = 120  # seconds; leave margin before 150s kill


def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_cycle(adj, start):
    """Find a simple cycle starting from 'start' using DFS. Returns list of vertices or None."""
    visited = {start: -1}
    parent = {start: None}
    stack = [start]
    found_cycle = None

    while stack:
        node = stack[-1]
        neighbors = sorted(adj[node])
        advanced = False
        for nb in neighbors:
            if nb not in visited:
                visited[nb] = len(stack)
                parent[nb] = node
                stack.append(nb)
                advanced = True
                break
        if not advanced:
            stack.pop()
            # Check if any neighbor leads back to an ancestor (forms a cycle)
            for nb in sorted(adj[node]):
                if nb in visited and nb != parent.get(node, None) and nb != start:
                    # Found a back edge to an ancestor
                    # Reconstruct cycle from nb to node
                    cycle = [node]
                    cur = node
                    while cur != nb:
                        cur = parent[cur]
                        cycle.append(cur)
                    found_cycle = cycle
                    break
            if found_cycle:
                break
    return found_cycle


def greedy_cycle_decomposition(edges, max_iters=5000):
    """
    Greedily extract cycles. At each step, find a cycle in the remaining graph,
    remove its edges, and repeat. Remaining edges become single-edge pieces.
    """
    adj = build_adjacency(edges)
    remaining_edges = set(edges)
    pieces = []

    for _ in range(max_iters):
        if not remaining_edges:
            break
        # Find a cycle
        cycle = None
        for start in sorted(adj.keys()):
            if not adj[start]:
                continue
            cycle = find_cycle(adj, start)
            if cycle:
                break
        if not cycle:
            break

        # Extract edges of the cycle
        cycle_edges = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in remaining_edges:
                cycle_edges.add(e)

        if len(cycle_edges) < 3:
            # Not a valid cycle in terms of all edges present
            # Try to find a proper cycle
            continue

        # Verify all cycle edges are in remaining
        if not all(e in remaining_edges for e in cycle_edges):
            continue

        pieces.append(cycle)
        for e in cycle_edges:
            remaining_edges.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)

    # Remaining edges as single pieces
    for e in sorted(remaining_edges):
        pieces.append([e[0], e[1]])

    return pieces


def greedy_longest_cycle_decomposition(edges, max_iters=5000):
    """
    Greedily extract the longest cycle found at each step.
    """
    adj = build_adjacency(edges)
    remaining_edges = set(edges)
    pieces = []

    for _ in range(max_iters):
        if not remaining_edges:
            break
        # Find the longest cycle
        best_cycle = None
        best_len = 0
        for start in sorted(adj.keys()):
            if not adj[start]:
                continue
            cycle = find_cycle(adj, start)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle

        if not best_cycle:
            break

        # Extract edges
        cycle_edges = set()
        for i in range(len(best_cycle)):
            u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = (min(u, v), max(u, v))
            if e in remaining_edges:
                cycle_edges.add(e)

        if len(cycle_edges) < 3:
            continue

        if not all(e in remaining_edges for e in cycle_edges):
            continue

        pieces.append(best_cycle)
        for e in cycle_edges:
            remaining_edges.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)

    for e in sorted(remaining_edges):
        pieces.append([e[0], e[1]])

    return pieces


def two_color_cycle_decomposition(edges, max_iters=5000):
    """
    Use a strategy that prefers cycles through high-degree vertices.
    """
    adj = build_adjacency(edges)
    remaining_edges = set(edges)
    pieces = []

    for _ in range(max_iters):
        if not remaining_edges:
            break
        # Start from highest-degree vertex
        best_cycle = None
        best_len = 0
        vertices_by_degree = sorted(adj.keys(), key=lambda v: -len(adj[v]))
        for start in vertices_by_degree:
            if not adj[start]:
                continue
            cycle = find_cycle(adj, start)
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
            if best_len >= 10:  # Good enough
                break

        if not best_cycle:
            break

        cycle_edges = set()
        for i in range(len(best_cycle)):
            u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = (min(u, v), max(u, v))
            if e in remaining_edges:
                cycle_edges.add(e)

        if len(cycle_edges) < 3:
            continue
        if not all(e in remaining_edges for e in cycle_edges):
            continue

        pieces.append(best_cycle)
        for e in cycle_edges:
            remaining_edges.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)

    for e in sorted(remaining_edges):
        pieces.append([e[0], e[1]])

    return pieces


def decompose_one(edges):
    """Try multiple strategies, pick the one with fewest pieces."""
    strategies = [
        greedy_cycle_decomposition,
        greedy_longest_cycle_decomposition,
        two_color_cycle_decomposition,
    ]
    best_pieces = None
    best_count = len(edges) + 1

    for strat in strategies:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        try:
            pieces = strat(edges)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
        except Exception:
            continue

    if best_pieces is None:
        # Fallback: all edges as single pieces
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            # Not enough time; fallback for remaining
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        pieces = decompose_one(edges)
        results.append(pieces)
    return results
