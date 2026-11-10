"""Clique partition of two fixed dense random graphs via bitmask-accelerated greedy with local improvement."""
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 138.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _build_adj_bitmasks(edges, n):
    """Build adjacency as bitmasks for fast intersection."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _greedy_partition_fast(edges, n, seed=0):
    """
    Greedy clique partition using bitmask operations for speed.
    Repeatedly extracts a large clique from remaining edges.
    """
    if not edges:
        return []

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    remaining = set(edge_set)
    cliques = []
    rng = _random.Random(seed)

    while remaining:
        if time.time() > _DEADLINE - 3:
            # Emergency: assign each remaining edge as its own clique
            for e in remaining:
                cliques.append([e[0], e[1]])
            break

        # Compute degrees in remaining graph
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1

        # Find vertex with max degree
        max_deg = max(deg)
        if max_deg == 0:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            continue

        candidates = [v for v in range(n) if deg[v] == max_deg]
        v = rng.choice(candidates) if len(candidates) > 1 else candidates[0]

        # Get neighbors of v in remaining graph
        S = set()
        for u, w in remaining:
            if u == v:
                S.add(w)
            elif w == v:
                S.add(u)

        if not S:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            continue

        # Build clique starting with v using bitmask intersection
        C = [v]
        # Bitmask of current candidate set
        cand_mask = 0
        for w in S:
            cand_mask |= (1 << w)

        while cand_mask:
            # Find vertex in candidate set with max degree within candidate set
            best_u = -1
            best_deg = -1
            best_candidates = []

            # Iterate over set bits in cand_mask
            m = cand_mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb

                # Count neighbors of u within cand_mask
                # Need adjacency restricted to remaining edges
                d = 0
                cm = cand_mask
                while cm:
                    lsb2 = cm & (-cm)
                    w = lsb2.bit_length() - 1
                    cm ^= lsb2
                    if u != w and (min(u, w), max(u, w)) in remaining:
                        d += 1

                if d > best_deg:
                    best_deg = d
                    best_u = u
                    best_candidates = [u]
                elif d == best_deg:
                    best_candidates.append(u)

            if best_u == -1:
                break

            if len(best_candidates) > 1:
                best_u = rng.choice(best_candidates)

            C.append(best_u)

            # Update candidate set: keep only vertices connected to best_u in remaining
            new_mask = 0
            cm = cand_mask
            while cm:
                lsb = cm & (-cm)
                w = lsb.bit_length() - 1
                cm ^= lsb
                if w != best_u and (min(best_u, w), max(best_u, w)) in remaining:
                    new_mask |= (1 << w)
            cand_mask = new_mask

        if len(C) >= 2:
            cliques.append(C)
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    e = (min(u, w), max(u, w))
                    remaining.discard(e)
        else:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))

    return cliques


def _local_improve(cliques, edges, n, max_iters=50):
    """
    Local improvement: try to merge pairs of cliques if their union is still a clique.
    Also try to move vertices between cliques to enable further merges.
    """
    if not cliques:
        return cliques

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    cliques = [list(c) for c in cliques]
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1

        # Try merging every pair of cliques
        i = 0
        while i < len(cliques):
            if not cliques[i]:
                cliques.pop(i)
                continue
            j = i + 1
            while j < len(cliques):
                # Check if union of cliques[i] and cliques[j] is a clique
                union = list(set(cliques[i] + cliques[j]))
                if len(union) >= 2:
                    is_clique = True
                    for a in range(len(union)):
                        for b in range(a + 1, len(union)):
                            u, w = union[a], union[b]
                            if (min(u, w), max(u, w)) not in edge_set:
                                is_clique = False
                                break
                        if not is_clique:
                            break

                    if is_clique:
                        cliques[i] = union
                        cliques.pop(j)
                        improved = True
                        # Don't increment j since we removed an element
                        continue
                j += 1
            i += 1

    return cliques


def _solve_instance(edges, n, time_budget=30.0):
    """Try multiple random restarts, keep best partition, then local improve."""
    best_cliques = None
    best_count = float('inf')
    start = time.time()

    # Deterministic first pass
    cliques = _greedy_partition_fast(edges, n, seed=0)
    best_cliques = cliques
    best_count = len(cliques)

    # Randomized restarts
    seed = 1
    while time.time() - start < time_budget:
        cliques = _greedy_partition_fast(edges, n, seed=seed)
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques
        seed += 1
        if seed > 200:
            break

    # Local improvement
    best_cliques = _local_improve(best_cliques, edges, n)

    return best_cliques


def construct():
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Allocate time: give more to the larger instance
    elapsed = time.time() - _T0
    remaining_time = _DEADLINE - _T0 - elapsed - 5  # 5s safety margin

    if remaining_time <= 0:
        res1 = _greedy_partition_fast(edges1, 32, seed=0)
        res2 = _greedy_partition_fast(edges2, 64, seed=0)
        return [res1, res2]

    # Split time: 40% for first, 60% for second (larger instance needs more)
    budget1 = min(remaining_time * 0.35, 25.0)
    budget2 = remaining_time - budget1 - 3

    res1 = _solve_instance(edges1, 32, time_budget=budget1)

    elapsed2 = time.time() - _T0
    if _DEADLINE - _T0 - elapsed2 < 5:
        res2 = _greedy_partition_fast(edges2, 64, seed=0)
    else:
        res2 = _solve_instance(edges2, 64, time_budget=max(budget2, 5.0))

    return [res1, res2]
