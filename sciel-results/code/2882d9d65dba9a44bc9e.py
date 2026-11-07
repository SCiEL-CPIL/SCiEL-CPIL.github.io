import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = 70.0  # safety margin before 90s kill

    # Build adjacency bitmasks over the FULL graph
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    vertices = list(range(n))

    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    def popcount(x):
        return bin(x).count('1')

    def is_valid_biclique(L_mask, R_mask):
        """Check every cross pair is a host edge, and sides disjoint/nonempty."""
        if L_mask == 0 or R_mask == 0:
            return False
        if (L_mask & R_mask) != 0:
            return False
        for u in get_bits(L_mask):
            if (adj[u] & R_mask) != R_mask:
                return False
        return True

    def build_maximal_biclique_from_seed(seed_L_mask, seed_R_mask, unc_adj, unc_edge_mask):
        """Grow (L,R) maximally w.r.t. uncovered edges while staying a valid biclique
        over the full graph. Returns (L_mask, R_mask) or None if seed invalid."""
        L = seed_L_mask
        R = seed_R_mask
        if (L & R) != 0 or L == 0 or R == 0:
            return None
        # Verify seed is a valid biclique in the full graph
        for u in get_bits(L):
            if (adj[u] & R) != R:
                return None
        changed = True
        while changed:
            changed = False
            # Candidate vertices that have uncovered edges
            cand_mask = 0
            for v in vertices:
                if unc_adj[v]:
                    cand_mask |= (1 << v)
            # Try adding to L: x not in L|R, x connects (in full graph) to all of R,
            # and x has at least one uncovered edge to R
            for x in get_bits(cand_mask & ~(L | R)):
                if (adj[x] & R) == R:
                    # ensure at least one uncovered cross edge exists
                    if (unc_adj[x] & R) != 0:
                        L |= (1 << x)
                        changed = True
            # Try adding to R: y not in L|R, y connects (full) to all of L
            for y in get_bits(cand_mask & ~(L | R)):
                if (adj[y] & L) == L:
                    if (unc_adj[y] & L) != 0:
                        R |= (1 << y)
                        changed = True
        return (L, R)

    def greedy_partition(max_left_size, seed_offset, use_maximal):
        unc_adj = adj[:]
        unc_edge_mask = 0
        for u, v in edges:
            unc_edge_mask |= (1 << u) | (1 << v)
        partition = []

        while True:
            if time.time() - start_time > deadline:
                return None
            # count remaining edges
            remaining = 0
            for u in vertices:
                if unc_adj[u]:
                    remaining += popcount(unc_adj[u])
            remaining //= 2
            if remaining == 0:
                break

            best_L = None
            best_R = None
            best_covered = 0
            best_rand = -1.0

            # candidate seeds: singletons and pairs
            for u in vertices:
                if unc_adj[u] == 0:
                    continue
                R_mask = unc_adj[u] & ~(1 << u)
                if R_mask == 0:
                    continue
                covered = popcount(R_mask)
                rv = random.random()
                if covered > best_covered or (covered == best_covered and rv > best_rand):
                    best_covered = covered
                    best_L = 1 << u
                    best_R = R_mask
                    best_rand = rv

            for i in range(n):
                for j in range(i + 1, n):
                    common = unc_adj[i] & unc_adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    if common == 0:
                        continue
                    covered = 2 * popcount(common)
                    rv = random.random()
                    if covered > best_covered or (covered == best_covered and rv > best_rand):
                        best_covered = covered
                        best_L = (1 << i) | (1 << j)
                        best_R = common
                        best_rand = rv

            if best_L is None:
                # fallback single edge
                for u in vertices:
                    m = unc_adj[u]
                    if m:
                        v = (m & -m).bit_length() - 1
                        best_L = 1 << u
                        best_R = 1 << v
                        break
                if best_L is None:
                    break

            if use_maximal:
                grown = build_maximal_biclique_from_seed(best_L, best_R, unc_adj, unc_edge_mask)
                if grown is not None:
                    best_L, best_R = grown

            # enforce disjoint & nonempty
            best_R &= ~best_L
            if best_R == 0:
                # should not happen; fallback
                m = unc_adj[get_bits(best_L)[0]] if best_L else 0
                if m:
                    v = (m & -m).bit_length() - 1
                    best_R = 1 << v
                else:
                    break

            partition.append((best_L, best_R))

            # remove covered edges
            for u in get_bits(best_L):
                unc_adj[u] &= ~best_R
            for v in get_bits(best_R):
                unc_adj[v] &= ~best_L

        return partition

    def try_merge(partition):
        """Local search: repeatedly try to merge two bicliques whose union is a valid
        biclique (disjoint sides, all cross pairs host edges)."""
        partition = [ (L, R) for (L, R) in partition ]
        changed = True
        while changed:
            changed = False
            for a in range(len(partition)):
                if time.time() - start_time > deadline:
                    break
                La, Ra = partition[a]
                for b in range(a + 1, len(partition)):
                    Lb, Rb = partition[b]
                    # To merge, we can put L = La ∪ Lb, R = Ra ∪ Rb only if that is a
                    # valid biclique AND the union of covered cross-edge sets equals the
                    # union of the two (no overlap, no gap). Since both are valid bicliques,
                    # union is valid iff every cross pair between (La∪Lb) and (Ra∪Rb) is a
                    # host edge AND the two bicliques are edge-disjoint (they are, by
                    # construction of a partition) AND no cross pair is "missing".
                    # Simplest safe merge: only when one side of b is subset-compatible.
                    # We attempt full union and validate.
                    L = La | Lb
                    R = Ra | Rb
                    if (L & R) != 0:
                        continue
                    if L == 0 or R == 0:
                        continue
                    # validate all cross pairs are host edges
                    ok = True
                    for u in get_bits(L):
                        if (adj[u] & R) != R:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Validate edge coverage: the union biclique must cover exactly the
                    # union of edges of the two original bicliques (no extra uncovered
                    # edges get absorbed that belong to other bicliques). We check that
                    # every cross edge of the union was covered by a or b.
                    # Build covered set of a and b
                    cov_a = 0
                    for u in get_bits(La):
                        cov_a |= (adj[u] & Ra)
                    cov_b = 0
                    for u in get_bits(Lb):
                        cov_b |= (adj[u] & Rb)
                    cov_union_edges = 0
                    for u in get_bits(L):
                        cov_union_edges |= (adj[u] & R)
                    if (cov_union_edges & ~(cov_a | cov_b)) != 0:
                        # union would cover edges not in a or b -> not allowed
                        continue
                    # also ensure no edge counted twice (partition property guarantees
                    # a and b are edge-disjoint, but double-check)
                    if (cov_a & cov_b) != 0:
                        continue
                    # merge
                    partition[a] = (L, R)
                    partition.pop(b)
                    changed = True
                    break
                if changed:
                    break
        return partition

    best_partition = None
    best_count = float('inf')

    # Run many randomized greedy attempts with varying maximal expansion
    attempts = 0
    while time.time() - start_time < deadline:
        random.seed(attempts)
        use_maximal = (attempts % 3 != 0)  # mix: 1/3 plain, 2/3 maximal
        part = greedy_partition(2, attempts, use_maximal)
        if part is None:
            attempts += 1
            continue
        # merge pass
        part2 = try_merge(part)
        if len(part2) < best_count:
            best_count = len(part2)
            best_partition = part2
        elif len(part) < best_count:
            best_count = len(part)
            best_partition = part
        attempts += 1
        if attempts > 400:
            break

    # Final fallback: individual edges
    if best_partition is None:
        result = [[[int(u)], [int(v)]] for u, v in edges]
        return result

    # Format output
    result = []
    for L, R in best_partition:
        left = [int(x) for x in get_bits(L)]
        right = [int(x) for x in get_bits(R)]
        result.append([left, right])

    return result
