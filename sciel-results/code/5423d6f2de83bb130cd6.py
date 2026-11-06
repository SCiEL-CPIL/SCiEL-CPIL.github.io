"""Rewritten program: greedy cycle decomposition to minimize piece count."""
import time
from random import Random

_DEADLINE = 140.0  # safety margin before 150s kill
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _find_longest_cycle(adj, start, max_len, visited_global):
    """DFS to find a long cycle starting from 'start'."""
    best = None
    stack = [(start, [start], {start})]
    found = 0
    while stack:
        node, path, on_path = stack.pop()
        found += 1
        if found > 200000:
            break
        neighbors = list(adj.get(node, ()))
        # sort neighbors to prefer longer paths
        for nb in neighbors:
            if nb == path[0] and len(path) >= 3:
                # found a cycle
                cyc = list(path)
                if best is None or len(cyc) > len(best):
                    best = cyc
                continue
            if nb in on_path:
                continue
            if nb in visited_global:
                continue
            if len(path) >= max_len:
                continue
            new_path = path + [nb]
            new_on = on_path | {nb}
            stack.append((nb, new_path, new_on))
    return best

def _find_any_cycle(adj, start):
    """DFS to find any cycle through start."""
    stack = [(start, [start], {start})]
    found = 0
    while stack:
        node, path, on_path = stack.pop()
        found += 1
        if found > 100000:
            break
        for nb in adj.get(node, ()):
            if nb == path[0] and len(path) >= 3:
                return list(path)
            if nb in on_path or nb in set():
                continue
            stack.append((nb, path + [nb], on_path | {nb}))
    return None

def _decompose(edges):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    edges_set = set(edges)
    adj = _build_adj(edges)
    pieces = []
    remaining = set(edges_set)

    # Try to find cycles greedily, preferring longer cycles.
    # We'll repeatedly scan for the best (longest) cycle we can find.
    deadline = _START + _DEADLINE
    max_iter = 500

    for _ in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if not remaining:
            break

        # Rebuild adjacency from remaining edges
        adj = {}
        for (u, v) in remaining:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Find the longest cycle by trying each vertex as start
        best_cycle = None
        for v in sorted(adj.keys()):
            if time.time() > deadline - 5.0:
                break
            cyc = _find_longest_cycle(adj, v, 50, set())
            if cyc is not None:
                if best_cycle is None or len(cyc) > len(best_cycle):
                    best_cycle = cyc
                    if len(best_cycle) >= 15:
                        break  # good enough

        if best_cycle is None:
            # Try simpler cycle finding
            for v in sorted(adj.keys()):
                if time.time() > deadline - 5.0:
                    break
                cyc = _find_any_cycle(adj, v)
                if cyc is not None:
                    best_cycle = cyc
                    break

        if best_cycle is None:
            break

        # Remove cycle edges from remaining
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % len(best_cycle)]
            e = (min(u, v), max(u, v))
            if e in remaining:
                remaining.discard(e)
            else:
                # Edge not found; cycle is invalid, skip
                best_cycle = None
                break

        if best_cycle is not None:
            pieces.append(best_cycle)

    # Leftover edges as single pieces
    for e in remaining:
        pieces.append([e[0], e[1]])

    return pieces

def construct():
    global _START
    _START = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        if time.time() > _START + _DEADLINE:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
    return results
