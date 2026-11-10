"""Cochromatic partition of a 14-vertex graph: minimize parts, each a clique or independent set."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    # adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    deadline = time.time() + 80.0
    best = None

    # Greedy first
    def greedy():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                ok = True
                for u in part:
                    if u < v:
                        e = (u & adj[v]) >> v & 1  # whether edge u-v exists
                    else:
                        e = (v & adj[u]) >> u & 1
                    # Determine if part is clique or independent set
                    # Check consistency: all pairs in part must agree on edge existence
                    break
                # Recompute: check if v can join part
                if ok:
                    # Determine type of part
                    if len(part) >= 2:
                        a, b = part[0], part[1]
                        edge_ab = (a & adj[b]) >> b & 1
                        for u in part:
                            if u == v:
                                continue
                            ev = (u & adj[v]) >> v & 1 if u > v else (v & adj[u]) >> u & 1
                            if edge_ab and not ev:
                                ok = False
                                break
                            if not edge_ab and ev:
                                ok = False
                                break
                    if ok:
                        part.append(v)
                        placed = True
                        break
            if not placed:
                parts.append([v])
        return parts

    best = greedy()

    # Try to improve with backtracking: try k from len(best)-1 down to 2
    # A partition into k parts means assigning each vertex to one of k "bins",
    # where each bin is either a clique or an independent set.
    # We can model this as: for each bin, a type (C or I) and a set of vertices.
    # Use backtracking with bitmasks.

    # Precompute for each vertex its non-neighbors (for independent set checks)
    # Actually, let's use a different approach:
    # A valid partition into k parts: assign each vertex a color 0..k-1.
    # For each color c, the induced subgraph on vertices of color c must be
    # either a clique or an independent set.
    #
    # We can do: for each subset S of vertices, check if S is a clique or independent set.
    # Then use set cover: find k subsets whose union is all vertices, disjoint, each clique/ind.
    #
    # Simpler: backtracking. Order vertices. For each vertex, try placing into existing
    # parts (checking compatibility) or opening a new part.
    # Prune: if current_parts + lower_bound(remaining) >= best_len, prune.

    # Lower bound: chromatic number of remaining graph (for clique parts)
    # or... actually the lower bound for cochromatic is max(chi(G), chi(complement G))
    # but that's for the whole graph. For a subset, we need a bound.
    # Simple lower bound: size of maximum clique in remaining (each clique vertex in different... no)
    # Actually each part can contain at most one independent set or one clique.
    # Lower bound: for the remaining vertices, the minimum number of parts needed.
    # A simple LB: max over all independent sets... no.
    # Let's use: LB = ceil(|remaining| / max_part_size) where max_part_size is the largest
    # clique or independent set in the remaining induced subgraph.
    # Or simpler: just use 1 per vertex as worst case, and rely on pruning by current count.

    # Better approach: use the fact that n=14 is small.
    # Enumerate all maximal cliques and maximal independent sets.
    # Then solve set cover: select minimum number of these sets that partition V.

    # Let's compute all cliques and independent sets (all of them, not just maximal).
    # There are at most 2^14 = 16384 subsets.

    all_sets = set()
    for mask in range(1, 1 << n):
        # Check if mask is a clique
        is_clique = True
        verts = [i for i in range(n) if mask & (1 << i)]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if not (adj[u] & (1 << v)):
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            all_sets.add((mask, 'C'))
            continue
        # Check if mask is an independent set
        is_ind = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if adj[u] & (1 << v):
                    is_ind = False
                    break
            if not is_ind:
                break
        if is_ind:
            all_sets.add((mask, 'I'))

    # Now we need to find a minimum-size collection of disjoint sets from all_sets
    # whose union is (1<<n)-1.
    # This is an exact set cover/partition problem.
    # Use DP over subsets: dp[mask] = min number of parts to cover 'mask' with valid sets.
    # But we need partition (disjoint union), so:
    # dp[mask] = min over submasks S of mask that are valid: 1 + dp[mask ^ S]
    # dp[0] = 0

    # n=14, 2^14 = 16384 states. For each state, iterate over submasks.
    # Number of submasks of a mask with k bits is 2^k. Total work ~ sum over masks of 2^{popcount(mask)} = 3^14 ~ 4.78M.
    # That's feasible.

    full = (1 << n) - 1
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    valid_masks = set(mask for mask, typ in all_sets)

    # For efficiency, precompute list of valid submasks for each mask?
    # Or iterate: for mask in range(1, 1<<n): for sub in submasks of mask: if sub in valid_masks: dp[mask] = min(dp[mask], 1+dp[mask^sub])

    # Iterate masks in increasing order of popcount to ensure sub-problems solved.
    # Actually dp[mask] depends on dp[mask ^ sub] where mask^sub < mask in terms of subset, so iterate by mask value works if we go from 0 to full.

    # To speed up, for each mask, iterate over valid submasks contained in mask.
    # Precompute for each valid sub, which masks contain it? No, just iterate submasks.

    # Optimization: for each mask, enumerate submasks of mask, check if in valid_masks.
    # Use the trick: sub = mask; while sub: check; sub = (sub-1) & mask

    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if sub in valid_masks:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    if dp[mask] <= len(best):
                        # Can't do better than current best minus remaining, but keep going
                        pass
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if dp[full] < len(best):
        # Reconstruct
        mask = full
        parts = []
        while mask:
            sub = mask
            while sub:
                if sub in valid_masks and dp[mask ^ sub] + 1 == dp[mask]:
                    parts.append([i for i in range(n) if sub & (1 << i)])
                    mask ^= sub
                    break
                sub = (sub - 1) & mask
            else:
                # Should not happen
                break
        if sum(len(p) for p in parts) == n:
            best = parts

    # Ensure return type: list of lists of int
    result = []
    for part in best:
        result.append([int(v) for v in part])
    return result
# EVOLVE-BLOCK-END
