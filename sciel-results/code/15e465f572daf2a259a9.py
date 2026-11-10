"""Clique partition of two fixed dense random graphs via optimized greedy with restarts."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _greedy_partition_fast(edges, n, rng):
    """
    Fast greedy clique partition using bitmasks for adjacency.
    """
    if not edges:
        return []

    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # Track which edges remain via a set of (min, max) tuples
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    cliques = []

    # Randomize vertex order for diversity
    vertex_order = list(range(n))
    rng.shuffle(vertex_order)

    # Precompute degree in remaining graph (updated dynamically)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    while remaining:
        if time.time() > _DEADLINE - 5:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                cliques.append([e[0], e[1]])
            break

        # Find vertex with max degree in remaining graph
        best_v = -1
        best_deg = -1
        for v in vertex_order:
            if deg[v] > best_deg:
                best_deg = deg[v]
                best_v = v
                if best_deg == 0:
                    break

        if best_v == -1 or best_deg == 0:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            deg[u] -= 1
            deg[w] -= 1
            continue

        # Build clique starting with best_v using greedy extension with bitmask
        C = [best_v]
        # Candidates: neighbors of best_v that have remaining edges
        cand_mask = adj_mask[best_v]
        # Filter to only vertices that actually have remaining edges with best_v
        for u in range(n):
            if u != best_v and (1 << u) & cand_mask:
                e = (min(best_v, u), max(best_v, u))
                if e not in remaining:
                    cand_mask &= ~(1 << u)

        # Greedily extend using bitmask operations
        while cand_mask:
            # Pick candidate with max remaining degree within candidates
            best_u = -1
            best_deg_u = -1
            temp_mask = cand_mask
            while temp_mask:
                u = (temp_mask & -temp_mask).bit_length() - 1
                temp_mask &= temp_mask - 1
                # Count connections to other candidates
                common = adj_mask[u] & cand_mask & ~(1 << u)
                d = bin(common).count('1')
                if d > best_deg_u:
                    best_deg_u = d
                    best_u = u

            if best_u == -1:
                break

            C.append(best_u)

            # Update candidates: keep only those connected to best_u in remaining
            new_cand_mask = cand_mask & adj_mask[best_u] & ~(1 << best_u)
            # Also ensure edges are still remaining
            for u in range(n):
                if (1 << u) & new_cand_mask:
                    e = (min(best_u, u), max(best_u, u))
                    if e not in remaining:
                        new_cand_mask &= ~(1 << u)
            cand_mask = new_cand_mask

        if len(C) >= 2:
            cliques.append(C)
            # Remove all edges within C from remaining and update degrees
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    e = (min(u, w), max(u, w))
                    if e in remaining:
                        remaining.discard(e)
                        deg[u] -= 1
                        deg[w] -= 1
        else:
            # Fallback: take one edge
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            deg[u] -= 1
            deg[w] -= 1

    return cliques


def _local_improve(cliques, edges, n, rng, max_iter=50):
    """
    Try to merge or restructure cliques to reduce count.
    Attempt: for each pair of cliques, check if their union is a clique.
    Also try moving vertices between cliques.
    """
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    cliques = [list(c) for c in cliques]
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if time.time() > _DEADLINE - 5:
            break
        
        # Try merging pairs of cliques
        new_cliques = []
        used = [False] * len(cliques)
        for i in range(len(cliques)):
            if used[i]:
                continue
            for j in range(i + 1, len(cliques)):
                if used[j]:
                    continue
                # Check if union of cliques[i] and cliques[j] is a clique
                union = set(cliques[i]) | set(cliques[j])
                is_clique = True
                for a in union:
                    for b in union:
                        if a >= b:
                            continue
                        if (a, b) not in edge_set:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique:
                    new_cliques.append(sorted(union))
                    used[i] = True
                    used[j] = True
                    improved = True
                    break
            if not used[i]:
                new_cliques.append(cliques[i])
        cliques = new_cliques
        
        # Try moving a vertex from one clique to another if it maintains clique property
        # and the source clique remains a clique
        for i in range(len(cliques)):
            if len(cliques[i]) < 3:
                continue
            for j in range(len(cliques)):
                if i == j or len(cliques[j]) < 2:
                    continue
                for v in cliques[i]:
                    # Check if v can be added to clique j
                    can_add = all((min(v, u), max(v, u)) in edge_set for u in cliques[j])
                    if can_add:
                        # Check if removing v from clique i still leaves a clique
                        remaining_i = [x for x in cliques[i] if x != v]
                        if len(remaining_i) >= 2:
                            is_still_clique = all(
                                (min(a, b), max(a, b)) in edge_set 
                                for a in remaining_i for b in remaining_i if a < b
                            )
                            if is_still_clique:
                                # Move v
                                cliques[i] = remaining_i
                                cliques[j] = cliques[j] + [v]
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
    
    # Filter out any cliques with < 2 vertices (shouldn't happen but safety)
    cliques = [c for c in cliques if len(c) >= 2]
    return cliques


def _validate_partition(cliques, edges, edge_set):
    """Validate that cliques form a valid partition of edges."""
    edge_count = 0
    seen_edges = set()
    for c in cliques:
        k = len(c)
        if k < 2:
            return False
        edge_count += k * (k - 1) // 2
        for i in range(k):
            for j in range(i + 1, k):
                e = (min(c[i], c[j]), max(c[i], c[j]))
                if e not in edge_set or e in seen_edges:
                    return False
                seen_edges.add(e)
    return edge_count == len(edges)


def _solve_instance(edges, n, max_restarts=None):
    """
    Solve a single instance with randomized restarts, keeping the best partition.
    """
    if not edges:
        return []

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)

    if max_restarts is None:
        # Estimate number of restarts based on remaining time
        remaining_time = _DEADLINE - 5 - time.time()
        if remaining_time < 3:
            max_restarts = 1
        elif n <= 32:
            max_restarts = min(100, int(remaining_time * 8))
        else:
            max_restarts = min(60, int(remaining_time * 5))

    best_cliques = None
    best_count = float('inf')

    for restart in range(max_restarts):
        if time.time() > _DEADLINE - 5:
            break

        rng = _random.Random(1000 + restart)
        cliques = _greedy_partition_fast(edges, n, rng)

        # Validate
        if not _validate_partition(cliques, edges, edge_set):
            continue

        # Apply local improvement
        cliques = _local_improve(cliques, edges, n, rng, max_iter=20)

        # Re-validate after improvement
        if not _validate_partition(cliques, edges, edge_set):
            continue

        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

    if best_cliques is None:
        # Fallback: single restart without validation
        rng = _random.Random(0)
        best_cliques = _greedy_partition_fast(edges, n, rng)

    return best_cliques


def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    # Solve smaller instance first to use time more efficiently
    res1 = _solve_instance(edges1, 32)
    res2 = _solve_instance(edges2, 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
