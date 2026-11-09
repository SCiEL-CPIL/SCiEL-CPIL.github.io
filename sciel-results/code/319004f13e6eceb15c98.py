"""Different strategy: direct biclique-merging local search."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    t0 = time.time()
    deadline = t0 + 85  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Host adjacency bitmask (for validating bicliques)
    host_adj = [0] * n
    for u, v in edges:
        host_adj[u] |= (1 << v)
        host_adj[v] |= (1 << u)

    # All edges as a frozenset of frozensets
    all_edges = set(frozenset((u, v)) for u, v in edges)

    def edges_of_biclique(left_mask, right_mask):
        """Return the set of edges (frozensets) in this biclique."""
        es = set()
        lm = left_mask
        while lm:
            u = (lm & -lm).bit_length() - 1
            lm &= lm - 1
            # u must connect to all in right_mask
            rm = right_mask
            while rm:
                v = (rm & -rm).bit_length() - 1
                rm &= rm - 1
                es.add(frozenset((u, v)))
        return es

    def is_valid_biclique(left_mask, right_mask):
        """Check that every cross pair is a host edge."""
        if not left_mask or not right_mask:
            return False
        lm = left_mask
        while lm:
            u = (lm & -lm).bit_length() - 1
            lm &= lm - 1
            # Check u connects to all in right_mask
            if (host_adj[u] & right_mask) != right_mask:
                return False
        return True

    def can_merge(bc1, bc2):
        """Try to merge two bicliques. Returns (new_left, new_right) or None."""
        l1, r1 = bc1
        l2, r2 = bc2
        # Orientation 1: left = l1|l2, right = r1&r2
        nl = l1 | l2
        nr = r1 & r2
        if nl and nr and is_valid_biclique(nl, nr):
            return (nl, nr)
        # Orientation 2: left = l1|r2, right = r1&l2
        nl = l1 | r2
        nr = r1 & l2
        if nl and nr and is_valid_biclique(nl, nr):
            return (nl, nr)
        # Orientation 3: left = l2|r1, right = r2&l1
        nl = l2 | r1
        nr = r2 & l1
        if nl and nr and is_valid_biclique(nl, nr):
            return (nl, nr)
        # Orientation 4: left = r1|r2, right = l1&l2
        nl = r1 | r2
        nr = l1 & l2
        if nl and nr and is_valid_biclique(nl, nr):
            return (nl, nr)
        return None

    # --- Phase 1: Build a pool of candidate bicliques ---
    # Start with single edges, then greedily add large bicliques found by expansion.

    # Greedy expansion to find large bicliques
    def find_large_bicliques():
        """Find a set of large bicliques via greedy expansion."""
        bicliques = []
        remaining = set(all_edges)

        while remaining:
            # Pick a random edge to start
            e = random.choice(list(remaining))
            u, v = tuple(e)

            left_mask = 1 << u
            right_mask = 1 << v

            # Expand greedily
            changed = True
            while changed:
                changed = False
                # Try adding to left
                for w in range(n):
                    if (left_mask >> w) & 1 or (right_mask >> w) & 1:
                        continue
                    # w must connect to all in right_mask
                    if (host_adj[w] & right_mask) != right_mask:
                        continue
                    # Also check that w's edges to right_mask are in remaining
                    # (not strictly needed since host edges are all in all_edges)
                    left_mask |= (1 << w)
                    changed = True
                # Try adding to right
                for w in range(n):
                    if (left_mask >> w) & 1 or (right_mask >> w) & 1:
                        continue
                    if (host_adj[w] & left_mask) != left_mask:
                        continue
                    right_mask |= (1 << w)
                    changed = True

            # Compute edges of this biclique
            es = edges_of_biclique(left_mask, right_mask)
            if es:
                bicliques.append((left_mask, right_mask))
                remaining -= es

        return bicliques

    # Generate multiple random pools and keep the best
    best_pool = None
    best_pool_count = float('inf')
    pool_count = 0

    while (time.time() - t0) < deadline - 10:  # leave 10s for local search
        random.seed()  # reseed for variety
        pool = find_large_bicliques()
        # Also add single edges for any uncovered
        covered = set()
        for lm, rm in pool:
            covered |= edges_of_biclique(lm, rm)
        for e in all_edges - covered:
            u, v = tuple(e)
            pool.append((1 << u, 1 << v))

        pool_count = len(pool)
        if pool_count < best_pool_count:
            best_pool_count = pool_count
            best_pool = pool

    if best_pool is None:
        # Fallback: single edges
        best_pool = [(1 << u, 1 << v) for u, v in edges]

    # --- Phase 2: Local search via biclique merging ---
    # We want to reduce the number of bicliques by merging.
    # A merge is valid if the merged biclique's edges are a subset of the union
    # of the two original bicliques' edges (i.e., no new edges are introduced).

    def pool_edges(bc):
        return edges_of_biclique(bc[0], bc[1])

    def total_edges(pool):
        covered = set()
        for bc in pool:
            covered |= pool_edges(bc)
        return covered

    # Verify initial pool covers all edges
    if total_edges(best_pool) != all_edges:
        # Add missing edges as single-edge bicliques
        covered = total_edges(best_pool)
        for e in all_edges - covered:
            u, v = tuple(e)
            best_pool.append((1 << u, 1 << v))

    current_pool = list(best_pool)
    current_count = len(current_pool)

    # Precompute edge sets for each biclique in current_pool
    bc_edges = [pool_edges(bc) for bc in current_pool]

    # Local search: repeatedly try to merge two bicliques
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        # Shuffle for variety
        indices = list(range(current_count))
        random.shuffle(indices)

        for i in indices:
            if i >= current_count:
                break
            for j in indices:
                if j >= current_count or j == i:
                    continue
                merged = can_merge(current_pool[i], current_pool[j])
                if merged is None:
                    continue
                nl, nr = merged
                merged_edges = edges_of_biclique(nl, nr)
                # Check that merged edges are subset of union of i and j
                if merged_edges <= (bc_edges[i] | bc_edges[j]):
                    # Perform the merge
                    # Remove j first (higher index to keep i stable)
                    if i < j:
                        del current_pool[j]
                        del bc_edges[j]
                        # Now i is still valid
                    else:
                        del current_pool[i]
                        del bc_edges[i]
                        # j becomes i
                        j = i

                    # Replace i with merged
                    current_pool[i] = (nl, nr)
                    bc_edges[i] = merged_edges
                    current_count -= 1
                    improved = True
                    break
            if improved:
                break

    # Also try single removals (if a biclique's edges are covered by others)
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        indices = list(range(current_count))
        random.shuffle(indices)
        for i in indices:
            if i >= current_count:
                break
            # Check if bc_edges[i] is covered by union of all others
            others = set()
            for k in range(current_count):
                if k != i:
                    others |= bc_edges[k]
            if bc_edges[i] <= others:
                del current_pool[i]
                del bc_edges[i]
                current_count -= 1
                improved = True
                break

    # --- Build final result ---
    result = []
    for lm, rm in current_pool:
        left = [i for i in range(n) if (lm >> i) & 1]
        right = [i for i in range(n) if (rm >> i) & 1]
        if left and right:
            result.append([left, right])

    # Type check: ensure Python ints
    for bc in result:
        bc[0] = [int(x) for x in bc[0]]
        bc[1] = [int(x) for x in bc[1]]

    return result
# EVOLVE-BLOCK-END
