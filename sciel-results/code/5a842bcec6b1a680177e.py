"""Cochromatic partition via branch-and-bound set cover over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def _maximal_sets(adj, n):
    """Return list of maximal homogeneous sets (each a clique or independent set)."""
    sets = []
    for v in range(n):
        # Candidate neighbors: for a clique, neighbors of v; for independent set, non-neighbors of v.
        cand_clique = adj[v]
        cand_indep = [u for u in range(n) if u != v and not adj[v][u]]
        # Maximal clique containing v: build greedily with backtracking to ensure maximality
        # Simpler: enumerate all maximal cliques via Bron-Kerbosch style but we just need maximal homogeneous sets.
        # We'll compute maximal cliques and maximal independent sets separately.
        pass
    # Bron-Kerbosch for maximal cliques
    def bronk(R, Pset, X):
        if not Pset and not X:
            sets.append(list(R))
        for v in list(Pset):
            bronk(R + [v], [u for u in Pset if adj[v][u]], [u for u in X if adj[v][u]])
            Pset.remove(v)
            X.append(v)
    bronk([], list(range(n)), [])
    # Maximal independent sets = maximal cliques of complement
    comp_adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            comp_adj[i][j] = (not adj[i][j]) and (i != j)
    def bronk2(R, Pset, X):
        if not Pset and not X:
            sets.append(list(R))
        for v in list(Pset):
            bronk2(R + [v], [u for u in Pset if comp_adj[v][u]], [u for u in X if comp_adj[v][u]])
            Pset.remove(v)
            X.append(v)
    bronk2([], list(range(n)), [])
    return sets

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin before 90s kill (evaluator 60s per contract, use margin)

    # Build adjacency bitmasks
    adj = [0]*n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Find all maximal homogeneous sets
    adj_bool = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj_bool[i][j] = bool(adj[i] >> j & 1)

    hom_sets = _maximal_sets(adj_bool, n)
    # Convert to bitmasks and deduplicate
    set_masks = []
    seen = set()
    for s in hom_sets:
        m = 0
        for v in s:
            m |= (1 << v)
        if m not in seen:
            seen.add(m)
            set_masks.append(m)

    full = (1 << n) - 1

    # Greedy initial upper bound: assign each vertex to a maximal set covering it
    # Simple greedy: repeatedly pick a set covering the most uncovered vertices
    uncovered = full
    chosen = []
    while uncovered:
        best = None
        best_cnt = -1
        for m in set_masks:
            c = bin(m & uncovered).count('1')
            if c > best_cnt:
                best_cnt = c
                best = m
        chosen.append(best)
        uncovered &= ~best
    ub = len(chosen)

    best_parts = []
    # Reconstruct parts from chosen masks (they are disjoint by construction of set cover)
    # Actually chosen may not be disjoint; let's just use a valid partition from greedy vertex assignment
    # Use the original greedy vertex assignment as a valid fallback
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    if len(parts) < ub:
        ub = len(parts)
    best_parts = [list(p) for p in parts]

    # Branch and bound over the set cover
    # Sort sets by size descending for better pruning
    set_masks_sorted = sorted(set_masks, key=lambda m: bin(m).count('1'), reverse=True)

    import sys
    sys.setrecursionlimit(10000)

    def bnb(remaining, depth, chosen_masks):
        nonlocal ub, best_parts
        if time.time() > deadline:
            return
        if not remaining:
            if depth < ub:
                ub = depth
                # Reconstruct partition: assign each vertex to a chosen set containing it
                # Chosen sets are disjoint in a valid set cover? Not necessarily.
                # For a valid cochromatic partition, parts must be disjoint and cover all.
                # Our chosen_masks are maximal homogeneous sets; if they overlap, we need to split.
                # To keep it simple and valid, we'll reconstruct by assigning each vertex to one set.
                assign = [-1]*n
                for idx, m in enumerate(chosen_masks):
                    for v in range(n):
                        if (m >> v) & 1 and assign[v] == -1:
                            assign[v] = idx
                # Check that each part (vertices assigned to idx) is homogeneous
                # Since each chosen set is homogeneous and we take subsets, subsets are homogeneous.
                parts_new = [[] for _ in range(len(chosen_masks))]
                for v in range(n):
                    parts_new[assign[v]].append(v)
                # Remove empty parts
                parts_new = [p for p in parts_new if p]
                if len(parts_new) <= depth:
                    best_parts = parts_new
            return
        # Lower bound: at least ceil(popcount(remaining)/max_set_size_in_remaining)
        # Simpler: at least 1 if remaining nonempty
        if depth + 1 >= ub:
            return
        # Pick a vertex in remaining to branch on
        # Choose vertex with fewest candidate sets covering it (MRV)
        rbits = remaining
        v = (rbits & -rbits).bit_length() - 1
        # Find sets that contain v and are subsets-compatible (all their vertices are in remaining)
        candidates = [m for m in set_masks_sorted if (m >> v) & 1 and (m & ~remaining) == 0]
        if not candidates:
            return
        # Try candidates
        for m in candidates:
            if time.time() > deadline:
                return
            new_remaining = remaining & ~m
            chosen_masks.append(m)
            bnb(new_remaining, depth + 1, chosen_masks)
            chosen_masks.pop()
            if depth + 1 >= ub:
                return

    bnb(full, 0, [])

    # Ensure best_parts is a valid partition
    # Validate
    if not best_parts:
        return [list(p) for p in parts]
    # Check all vertices covered exactly once
    seen = set()
    for part in best_parts:
        for v in part:
            if v in seen:
                # Fallback to valid greedy
                return [list(p) for p in parts]
            seen.add(v)
    if seen != set(range(n)):
        return [list(p) for p in parts]
    # Check each part homogeneous
    for part in best_parts:
        if len(part) == 1:
            continue
        hits = []
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                hits.append((part[i], part[j]) in es or (part[j], part[i]) in es)
        if not (all(hits) or not any(hits)):
            return [list(p) for p in parts]
    # Return as list of lists of ints
    return [[int(v) for v in part] for part in best_parts]
# EVOLVE-BLOCK-END
