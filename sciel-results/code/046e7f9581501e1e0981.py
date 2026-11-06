"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _greedy_decompose(edges, n, rng, time_limit):
    """
    Initial greedy decomposition: repeatedly find a cycle (randomized DFS) and remove it.
    """
    if not edges:
        return []

    start_time = time.time()
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    pieces = []

    while remaining:
        if time.time() - start_time > time_limit:
            break

        best_cycle = None
        best_len = 0

        for start in range(n):
            if time.time() - start_time > time_limit:
                break
            deg = sum(1 for w in adj[start] if (min(start, w), max(start, w)) in remaining)
            if deg < 2:
                continue

            # Randomized DFS to find a cycle
            neighbors = [w for w in adj[start]
                         if (min(start, w), max(start, w)) in remaining and w > start]
            if len(neighbors) < 2:
                continue

            # Try a few random orderings
            for _ in range(3):
                rng.shuffle(neighbors)
                path = [start]
                visited = {start}

                def dfs(u):
                    for v in neighbors if u == start else \
                            [w for w in adj[u] if (min(u, w), max(u, w)) in remaining and w >= start]:
                        if v == start:
                            if len(path) >= 3:
                                return path[:]
                            continue
                        if v < start:
                            continue
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            result = dfs(v)
                            if result:
                                return result
                            path.pop()
                            visited.discard(v)
                    return None

                cycle = dfs(start)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    break

        if best_cycle:
            valid = all((min(best_cycle[i], best_cycle[(i+1)%len(best_cycle)]),
                        max(best_cycle[i], best_cycle[(i+1)%len(best_cycle)])) in remaining
                        for i in range(len(best_cycle)))
            if valid:
                pieces.append(best_cycle)
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i+1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                continue
        else:
            for e in remaining:
                pieces.append(list(e))
            remaining = set()
            break

    for e in remaining:
        pieces.append(list(e))

    return pieces


def _local_search(pieces, edges_set, n, time_limit):
    """
    Local search: try to merge two single-edge pieces into a cycle if possible,
    or rearrange pieces to reduce total count.
    """
    start_time = time.time()
    rng = random.Random(123)

    # Count pieces
    def count_pieces(p):
        return len(p)

    best_pieces = pieces[:]
    best_count = count_pieces(best_pieces)

    while time.time() - start_time < time_limit:
        improved = False

        # Try merging two edges into a 2-edge path... no, that doesn't help.
        # Instead: try to find a cycle among currently-unpaired edges
        # that weren't in the original decomposition

        # Get all edges that are single-edge pieces
        single_edges = []
        for i, piece in enumerate(best_pieces):
            if len(piece) == 2:
                single_edges.append((i, (min(piece[0], piece[1]), max(piece[0], piece[1]))))

        if len(single_edges) >= 3:
            # Try to find a cycle among these single edges
            # Build subgraph
            sub_adj = defaultdict(set)
            for idx, e in single_edges:
                u, v = e
                sub_adj[u].add(v)
                sub_adj[v].add(u)

            # Find a cycle in this subgraph
            for start in sub_adj:
                if len(sub_adj[start]) < 2:
                    continue
                # DFS
                path = [start]
                visited = {start}

                def dfs(u):
                    for v in sub_adj[u]:
                        if v == start:
                            if len(path) >= 3:
                                return path[:]
                            continue
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            result = dfs(v)
                            if result:
                                return result
                            path.pop()
                            visited.discard(v)
                    return None

                cycle = dfs(start)
                if cycle and len(cycle) >= 3:
                    # Replace the single edges in this cycle with one cycle piece
                    cycle_edges = set()
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i+1) % len(cycle)]
                        cycle_edges.add((min(u, v), max(u, v)))

                    # Remove single-edge pieces that are in cycle_edges
                    new_pieces = []
                    removed_count = 0
                    for piece in best_pieces:
                        if len(piece) == 2:
                            e = (min(piece[0], piece[1]), max(piece[0], piece[1]))
                            if e in cycle_edges:
                                removed_count += 1
                                continue
                        new_pieces.append(piece)

                    if removed_count >= 3:
                        new_pieces.append(cycle)
                        new_count = count_pieces(new_pieces)
                        if new_count < best_count:
                            best_pieces = new_pieces
                            best_count = new_count
                            improved = True
                            break

        if not improved:
            # Try random perturbation: swap a cycle for its edges and re-greedy
            if best_count > 5:
                # Pick a random cycle piece and break it into edges
                cycle_indices = [i for i, p in enumerate(best_pieces) if len(p) >= 3]
                if cycle_indices:
                    idx = rng.choice(cycle_indices)
                    cycle = best_pieces[idx]
                    # Replace with individual edges
                    new_pieces = []
                    for i, piece in enumerate(best_pieces):
                        if i == idx:
                            for j in range(len(cycle)):
                                u = cycle[j]
                                v = cycle[(j+1) % len(cycle)]
                                new_pieces.append([u, v])
                        else:
                            new_pieces.append(piece)

                    # Now try to re-find cycles among these edges
                    # This is expensive, so just accept if we can find at least one
                    # For now, skip this perturbation
                    pass

        if not improved:
            break

    return best_pieces


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
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        rng = random.Random(seed + 1000)

        # Phase 1: Greedy decomposition
        greedy_time = min(remaining_time * 0.4, 20.0)
        pieces = _greedy_decompose(edges, n, rng, greedy_time)

        # Phase 2: Local search to improve
        ls_time = min(remaining_time * 0.5, 25.0)
        pieces = _local_search(pieces, edges_set, n, ls_time)

        results.append(pieces)

    return results
