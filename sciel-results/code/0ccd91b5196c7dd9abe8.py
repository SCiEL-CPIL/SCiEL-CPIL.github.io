import time
import random

def construct():
    start_time = time.time()
    HARD_DEADLINE = start_time + 140.0  # Safety margin before 150s limit
    n = 20

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    def popcount(x):
        return bin(x).count('1')

    def _greedy_construct(seed):
        """
        Fast greedy construction using bitmask adjacency.
        Returns list of edges [u, v].
        """
        random.seed(seed)
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)

        adj_mask = [0] * n  # bitmask of neighbors
        current_edges = []

        for u, v in edges_to_try:
            # Check: common neighbors of u and v must be < 3
            common = popcount(adj_mask[u] & adj_mask[v])
            if common >= 3:
                continue

            # Check: for every neighbor w of u (w != v), common(w, v) must be < 2
            # because adding (u,v) increases common(w,v) by 1
            ok = True
            w = adj_mask[u]
            w &= ~(1 << v)
            while w:
                low = w & (-w)
                w -= low
                wi = low.bit_length() - 1
                if popcount(adj_mask[wi] & adj_mask[v]) >= 2:
                    ok = False
                    break
            if not ok:
                continue

            # Check: for every neighbor w of v (w != u), common(u, w) must be < 2
            w = adj_mask[v]
            w &= ~(1 << u)
            while w:
                low = w & (-w)
                w -= low
                wi = low.bit_length() - 1
                if popcount(adj_mask[u] & adj_mask[wi]) >= 2:
                    ok = False
                    break
            if not ok:
                continue

            # Add edge
            current_edges.append((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        return current_edges

    def _local_search(initial_edges, seed, time_limit):
        """
        Local search starting from initial_edges using bitmasks.
        Tries systematic fill; if stuck, removes an edge that blocks many additions.
        """
        random.seed(seed)

        adj_mask = [0] * n
        edges = list(initial_edges)
        edge_set = set(edges)

        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        current_count = len(edges)
        best_local_count = current_count
        best_local_edges = [e[:] for e in edges]

        def can_add(u, v):
            if edge_set and (u, v) in edge_set:
                return False
            if (v, u) in edge_set:
                return False
            # Common neighbors of u and v
            if popcount(adj_mask[u] & adj_mask[v]) >= 3:
                return False
            # For each neighbor w of u (w != v), common(w, v) < 2
            w = adj_mask[u] & ~(1 << v)
            while w:
                low = w & (-w)
                w -= low
                wi = low.bit_length() - 1
                if popcount(adj_mask[wi] & adj_mask[v]) >= 2:
                    return False
            # For each neighbor w of v (w != u), common(u, w) < 2
            w = adj_mask[v] & ~(1 << u)
            while w:
                low = w & (-w)
                w -= low
                wi = low.bit_length() - 1
                if popcount(adj_mask[u] & adj_mask[wi]) >= 2:
                    return False
            return True

        def do_add(u, v):
            edges.append((u, v))
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        def do_remove(u, v):
            edges.remove((u, v))
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)

        def systematic_fill():
            """Try to add all missing edges that are valid, in order."""
            nonlocal current_count, best_local_count, best_local_edges
            added_any = False
            for u, v in all_edges:
                if (u, v) not in edge_set and can_add(u, v):
                    do_add(u, v)
                    current_count += 1
                    added_any = True
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = [e[:] for e in edges]
            return added_any

        def count_blocked(u, v):
            """Count how many missing edges are blocked by edge (u,v)."""
            if (u, v) not in edge_set:
                return 0
            # Temporarily remove
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            edge_set.discard((u, v))

            blocked = 0
            for a, b in all_edges:
                if (a, b) in edge_set:
                    continue
                if can_add(a, b):
                    blocked += 1

            # Restore
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add((u, v))
            return blocked

        # Local search loop
        for it in range(20000):
            if time.time() > time_limit:
                break

            # First, try systematic fill
            if systematic_fill():
                continue

            # Stuck: try removing an edge that blocks the most additions
            if edges:
                # Sample a few edges to find one that blocks many additions
                best_remove = None
                best_blocked = -1
                sample_size = min(len(edges), 15)
                sampled = random.sample(edges, sample_size)
                for e in sampled:
                    b = count_blocked(e[0], e[1])
                    if b > best_blocked:
                        best_blocked = b
                        best_remove = e

                if best_remove is not None:
                    do_remove(best_remove[0], best_remove[1])
                    current_count -= 1
                else:
                    # Fallback: remove random edge
                    e = random.choice(edges)
                    do_remove(e[0], e[1])
                    current_count -= 1

        return best_local_edges, best_local_count

    # Phase 1: Run many greedy constructions to find good starting points
    greedy_results = []
    best_count = 0
    best_edges = []

    num_greedy = 400
    for seed in range(num_greedy):
        if time.time() > HARD_DEADLINE - 60.0:
            break
        result = _greedy_construct(seed)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result[:]
        greedy_results.append((len(result), result))

    # Phase 2: Local search from the best greedy results
    greedy_results.sort(key=lambda x: -x[0])

    # Run local search on top 12 greedy results
    for i in range(min(12, len(greedy_results))):
        if time.time() > HARD_DEADLINE - 15.0:
            break
        _, start_edges = greedy_results[i]
        ls_edges, ls_count = _local_search(start_edges, seed=10000 + i, time_limit=HARD_DEADLINE - 15.0)
        if ls_count > best_count:
            best_count = ls_count
            best_edges = ls_edges[:]

    # Phase 3: Additional local search passes on the best solution found
    if best_count > 0 and time.time() < HARD_DEADLINE - 10.0:
        for extra_seed in range(8):
            if time.time() > HARD_DEADLINE - 5.0:
                break
            ls_edges, ls_count = _local_search(best_edges, seed=50000 + extra_seed, time_limit=HARD_DEADLINE - 5.0)
            if ls_count > best_count:
                best_count = ls_count
                best_edges = ls_edges[:]

    # Phase 4: Final polish - one more local search with more time
    if best_count > 0 and time.time() < HARD_DEADLINE - 3.0:
        ls_edges, ls_count = _local_search(best_edges, seed=99999, time_limit=HARD_DEADLINE - 1.0)
        if ls_count > best_count:
            best_count = ls_count
            best_edges = ls_edges[:]

    # Convert to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
