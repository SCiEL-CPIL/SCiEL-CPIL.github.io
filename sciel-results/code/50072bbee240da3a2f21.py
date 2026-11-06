"""Clique partition of two fixed dense random graphs: greedy maximal clique packing."""
import time

_DEADLINE = 135.0  # wall-clock safety margin below the 150s kill
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge set for (n, p, seed)."""
    import random
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _validate(n, edges, cliques):
    """Every listed set is a clique (>=2) and every edge lies in exactly one clique."""
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set


def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    edge_set = set(edges)
    if not edges:
        return []

    # Adjacency bitsets (int masks)
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute triangle counts per edge for ordering
    edge_tri_count = {}
    for (u, v) in edges:
        if u > v:
            u, v = v, u
        cand = adj[u] & adj[v]
        cand &= ~((1 << (v + 1)) - 1)  # only w > v
        edge_tri_count[(u, v)] = cand.bit_count()

    # Sort edges: most constrained (fewest triangle options) first
    sorted_edges = sorted(edges, key=lambda e: edge_tri_count.get((min(e[0], e[1]), max(e[0], e[1])), 0))

    used = set()
    cliques = []

    for (u, v) in sorted_edges:
        if _elapsed() > _DEADLINE:
            break
        if (u, v) in used:
            continue

        # Build a maximal clique starting from edge {u, v}
        # Use only vertices > max(u, v) to avoid reprocessing
        max_uv = max(u, v)
        current_clique = [u, v]
        # Candidates: common neighbors of u and v, all > max_uv
        cand = adj[u] & adj[v]
        cand &= ~((1 << (max_uv + 1)) - 1)

        # Greedily extend: pick candidates that maintain clique property
        # Order candidates by degree (lower degree first for more constrained)
        while cand and _elapsed() < _DEADLINE:
            # Pick the candidate with lowest degree among remaining
            best_w = -1
            best_deg = float('inf')
            temp = cand
            while temp:
                w = (temp & -temp).bit_length() - 1
                deg = adj[w].bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_w = w
                temp &= temp - 1

            if best_w < 0:
                break

            # Check if best_w is adjacent to all current clique members
            can_add = True
            for x in current_clique:
                if not (adj[best_w] >> x) & 1:
                    can_add = False
                    break

            if can_add:
                current_clique.append(best_w)
                # Update candidates: must be adjacent to all clique members
                cand = cand & adj[best_w]
                new_max = max(current_clique)
                cand &= ~((1 << (new_max + 1)) - 1)
            else:
                # Remove this candidate and try next
                cand &= ~(1 << best_w)

        # Mark all edges in this clique as used
        k = len(current_clique)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = current_clique[i], current_clique[j]
                e = (a, b) if a < b else (b, a)
                used.add(e)

        cliques.append(current_clique)

    # Remaining edges become single-edge cliques
    for (u, v) in edges:
        if (u, v) not in used:
            cliques.append([u, v])

    # Safety: verify the partition; fall back to all single-edge cliques if invalid
    if not _validate(n, edges, cliques):
        cliques = [[u, v] for (u, v) in edges]
    return cliques


def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
