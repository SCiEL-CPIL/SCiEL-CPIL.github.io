"""Cochromatic partition via branch-and-bound over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute for each vertex: compatible vertices for clique (neighbors) and independent set (non-neighbors)
    # clique_comp[v] = bitmask of vertices that can join a clique with v (i.e., neighbors of v)
    # indep_comp[v] = bitmask of vertices that can join an independent set with v (i.e., non-neighbors of v)
    clique_comp = [0] * n
    indep_comp = [0] * n
    for v in range(n):
        clique_comp[v] = adj[v] & ((1 << v) - 1)  # neighbors (excluding self, but self not in adj)
        # Actually adj[v] doesn't include v, so:
        clique_comp[v] = adj[v]
        indep_comp[v] = ((1 << n) - 1) ^ (1 << v) ^ adj[v]  # all except self and neighbors

    # Enumerate all maximal cliques
    def find_maximal_cliques():
        cliques = []
        # Bron-Kerbosch with pivot
        def bronk(R, P_set, X):
            if P_set == 0 and X == 0:
                cliques.append(R)
                return
            # Choose pivot u from P_set | X
            union = P_set | X
            if union == 0:
                return
            # Pick pivot with max degree in union
            u = -1
            max_deg = -1
            temp = union
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj[v] & union).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    u = v
                temp ^= lsb
            # For each v in P_set \ N(u)
            candidates = P_set & ~adj[u]
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                bronk(R | lsb, P_set & adj[v], X & adj[v])
                P_set ^= lsb
                X |= lsb
                candidates ^= lsb
        bronk(0, (1 << n) - 1, 0)
        return cliques

    # Enumerate all maximal independent sets (maximal cliques in complement)
    def find_maximal_indep_sets():
        indep_sets = []
        def bronk(R, P_set, X):
            if P_set == 0 and X == 0:
                indep_sets.append(R)
                return
            union = P_set | X
            if union == 0:
                return
            u = -1
            max_deg = -1
            temp = union
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (indep_comp[v] & union).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    u = v
                temp ^= lsb
            candidates = P_set & ~indep_comp[u]
            while candidates:
                lsb = candidates & (-candidates)
                v = lsb.bit_length() - 1
                bronk(R | lsb, P_set & indep_comp[v], X & indep_comp[v])
                P_set ^= lsb
                X |= lsb
                candidates ^= lsb
        bronk(0, (1 << n) - 1, 0)
        return indep_sets

    max_cliques = find_maximal_cliques()
    max_indeps = find_maximal_indep_sets()

    # All candidate sets (cliques and independent sets)
    all_sets = [(c, 'clique') for c in max_cliques] + [(s, 'indep') for s in max_indeps]

    # For efficiency, also include all subsets? No, just maximal ones should suffice for minimum cover
    # Actually we need all homogeneous sets, not just maximal, because a non-maximal set might be needed
    # But for minimum set cover, using only maximal sets is sufficient (any solution using a non-maximal set
    # can be extended to use a maximal set containing it, which only helps)
    # Wait, that's not quite right - we need the sets to be disjoint. Let me reconsider.
    # Actually for this problem, we need disjoint homogeneous sets. So we should use all homogeneous sets.
    # But that's exponential. Let's use maximal ones and see if it works.

    # Convert to list of bitmasks for the sets
    set_masks = [s for s, _ in all_sets]
    set_types = [t for _, t in all_sets]

    # Remove dominated sets: if set A is a subset of set B and both are same type, A is dominated
    # Actually for disjoint cover, a smaller set might be needed. Let's keep all for now.
    # But filter: if a set is a proper subset of another set of the same type, the smaller is never needed
    # in an optimal solution (we can always extend it). So filter out non-maximal.
    # They're already maximal by construction.

    # Precompute for each set: which vertices it covers
    # Already have as bitmasks.

    # Branch and bound
    best_parts = None
    best_count = n + 1  # upper bound: each vertex in its own part

    # Sort sets by size descending for better pruning
    order = sorted(range(len(set_masks)), key=lambda i: -set_masks[i].bit_count())

    def popcount(x):
        return x.bit_count()

    def solve(remaining, current_parts, count):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return
        if count >= best_count:
            return
        if remaining == 0:
            best_parts = current_parts[:]
            best_count = count
            return

        # Lower bound: ceil(remaining.bit_count() / max_set_size_in_remaining)
        # More useful: for each vertex in remaining, it must be covered by some set
        # Simple LB: at least 1 more part needed
        # Better: greedy lower bound
        rem = remaining
        lb = 0
        temp_rem = remaining
        while temp_rem:
            # Find largest set that fits in temp_rem
            best_size = 0
            for idx in order:
                s = set_masks[idx]
                if (s & temp_rem) == s:  # set is subset of temp_rem
                    sz = s.bit_count()
                    if sz > best_size:
                        best_size = sz
                    break  # Since sorted by size, first fit is largest
            if best_size == 0:
                lb += 1
                # Remove one vertex (arbitrary)
                lsb = temp_rem & (-temp_rem)
                temp_rem ^= lsb
            else:
                lb += 1
                # Remove the best fitting set
                for idx in order:
                    s = set_masks[idx]
                    if (s & temp_rem) == s:
                        temp_rem &= ~s
                        break
        if count + lb >= best_count:
            return

        # Pick a vertex to cover - choose one with fewest options
        # Find vertex in remaining with minimum number of sets that can cover it
        best_v = -1
        min_options = n + 1
        temp = remaining
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            # Count sets that contain v and are subset of remaining
            cnt = 0
            for idx in order:
                s = set_masks[idx]
                if (s & lsb) and (s & remaining) == s:
                    cnt += 1
                    if cnt >= min_options:
                        break
            if cnt < min_options:
                min_options = cnt
                best_v = v
                if cnt <= 1:
                    break
            temp ^= lsb

        if best_v == -1:
            return

        v_bit = 1 << best_v

        # Try each set that contains best_v and is subset of remaining
        tried = set()
        for idx in order:
            s = set_masks[idx]
            if (s & v_bit) and (s & remaining) == s:
                if idx in tried:
                    continue
                tried.add(idx)
                new_remaining = remaining & ~s
                current_parts.append(list(range(n)))  # placeholder
                # Build the actual list of vertices in s
                verts = []
                temp_s = s
                while temp_s:
                    lsb = temp_s & (-temp_s)
                    verts.append(lsb.bit_length() - 1)
                    temp_s ^= lsb
                current_parts[-1] = verts
                solve(new_remaining, current_parts, count + 1)
                current_parts.pop()

    full_mask = (1 << n) - 1
    solve(full_mask, [], 0)

    if best_parts is None:
        # Fallback: greedy
        parts = []
        es = {tuple(e) for e in P['edges']}
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

    # Convert to proper lists of ints
    result = []
    for part in best_parts:
        result.append([int(v) for v in part])
    return result

# EVOLVE-BLOCK-END
