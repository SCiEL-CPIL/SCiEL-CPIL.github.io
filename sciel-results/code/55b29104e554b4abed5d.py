import time
import random
import math

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    # Build adjacency as bitmask
    adj = [0] * n
    edge_set = set()
    edge_list = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edge_list.append((u, v))

    m = len(edge_list)

    # Build edge index map: (u,v) -> index
    edge_idx = {}
    for i, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = i

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def get_random_edge(rem_mask):
        candidates = [u for u in range(n) if rem_mask[u] != 0]
        if not candidates:
            return None
        u = random.choice(candidates)
        mask = rem_mask[u]
        vs = []
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            vs.append(v)
            mm ^= lsb
        if not vs:
            return None
        v = random.choice(vs)
        if u > v:
            u, v = v, u
        return (u, v)

    def get_deterministic_edge(rem_mask, edge_idx_val):
        count = -1
        for u in range(n):
            mask = rem_mask[u]
            mm = mask
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                mm ^= lsb
                if u < v:
                    count += 1
                    if count == edge_idx_val:
                        return (u, v)
        return None

    def expand_biclique(L_mask, R_mask, rem_mask):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                if (L_mask >> u) & 1:
                    continue
                if R_mask != 0 and (rem_mask[u] & R_mask) == R_mask:
                    L_mask |= (1 << u)
                    changed = True
            for v in range(n):
                if (R_mask >> v) & 1:
                    continue
                if L_mask != 0 and (rem_mask[v] & L_mask) == L_mask:
                    R_mask |= (1 << v)
                    changed = True
        return L_mask, R_mask

    def try_greedy(restarts=300, seed=None):
        best_bicliques = None
        best_count = float('inf')
        if seed is not None:
            random.seed(seed)

        for r in range(restarts):
            if time.time() > deadline:
                break

            rem_mask = adj[:]
            edges_left = m
            current_bicliques = []

            use_deterministic = (r < 20)
            edge_counter = 0

            while edges_left > 0:
                if time.time() > deadline:
                    break

                best_local = None
                best_size = 0

                num_trials = 10 if use_deterministic else 6
                for trial in range(num_trials):
                    if time.time() > deadline:
                        break

                    if use_deterministic and trial < 4:
                        edge = get_deterministic_edge(rem_mask, edge_counter)
                        if edge is None:
                            break
                    else:
                        edge = get_random_edge(rem_mask)
                        if edge is None:
                            break

                    u, v = edge
                    L_mask = (1 << u)
                    R_mask = (1 << v)
                    L_mask, R_mask = expand_biclique(L_mask, R_mask, rem_mask)

                    size = L_mask.bit_count() * R_mask.bit_count()
                    if size > best_size:
                        best_size = size
                        best_local = (L_mask, R_mask)

                if not best_local:
                    edge = get_random_edge(rem_mask)
                    if edge is None:
                        break
                    u, v = edge
                    best_local = ((1 << u), (1 << v))
                    best_size = 1

                L_mask, R_mask = best_local
                current_bicliques.append([mask_to_list(L_mask), mask_to_list(R_mask)])

                l_list = mask_to_list(L_mask)
                r_list = mask_to_list(R_mask)
                for u in l_list:
                    rem_mask[u] &= ~R_mask
                for v in r_list:
                    rem_mask[v] &= ~L_mask

                edges_left -= best_size
                edge_counter += 1

                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    break

        return best_bicliques, best_count

    # Phase 1: Greedy with many restarts to get an initial upper bound
    best_bicliques, best_count = try_greedy(restarts=500)

    if best_bicliques is None:
        result = []
        for u in range(n):
            right_side = [v for u2, v in edges if u2 == u]
            if right_side:
                result.append([[u], right_side])
        return result

    # Phase 2: Enumerate all maximal bicliques using Bron-Kerbosch style on the bipartite complement
    # A biclique (L,R) is maximal if no vertex can be added to L or R while maintaining the complete bipartite property
    # We enumerate by: for each edge (u,v), grow to a maximal biclique containing that edge
    # Use a visited set to avoid duplicates

    # First, let's collect all maximal bicliques
    # Key insight: a biclique is determined by (L_mask, R_mask). Two bicliques are the same if they have the same L and R.
    # We'll use a set of (frozenset(L), frozenset(R)) to deduplicate.

    # For efficiency, we enumerate maximal bicliques by iterating over edges and expanding
    # But we need to ensure we get ALL maximal bicliques, not just those found by edge-based expansion

    # Alternative approach: enumerate all maximal bicliques using the following method:
    # 1. For each vertex u, consider all subsets of N(u) as potential right sides
    # 2. For each such subset R, the left side is {u} union {v : N(v) ⊇ R}
    # This is exponential but n=30 might be too large

    # Better: Use the fact that we only need bicliques that are "useful" for set cover
    # A biclique is useful if it covers at least one edge that no other single biclique covers
    # But this is hard to determine a priori

    # Let's use a hybrid: enumerate maximal bicliques via edge-based expansion with dedup
    # and also include the greedy-found bicliques

    maximal_bicliques = []  # list of (L_list, R_list)
    seen_bicliques = set()

    def add_biclique(L_mask, R_mask):
        L_list = mask_to_list(L_mask)
        R_list = mask_to_list(R_mask)
        # Normalize: sort both sides
        key = (tuple(sorted(L_list)), tuple(sorted(R_list)))
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            maximal_bicliques.append((L_list, R_list))

    # Enumerate maximal bicliques by expanding from each edge
    for (u, v) in edge_list:
        if time.time() > deadline - 5:
            break
        L_mask = (1 << u)
        R_mask = (1 << v)
        L_mask, R_mask = expand_biclique(L_mask, R_mask, adj)
        add_biclique(L_mask, R_mask)

    # Also try expanding from each vertex pair that forms a biclique of size 1x1
    # Already covered by edges above

    # Add the greedy-found bicliques as candidates too
    for b in best_bicliques:
        L_mask = 0
        for x in b[0]:
            L_mask |= (1 << x)
        R_mask = 0
        for x in b[1]:
            R_mask |= (1 << x)
        add_biclique(L_mask, R_mask)

    # Also add single-edge bicliques as fallback candidates
    for (u, v) in edge_list:
        add_biclique((1 << u), (1 << v))

    # Now we have candidate bicliques. Build the set cover problem.
    # Each biclique covers a set of edges.
    # We want to find the minimum number of bicliques that cover all edges.

    # Precompute the edge set covered by each biclique as a bitmask
    num_candidates = len(maximal_bicliques)
    biclique_edge_masks = []  # list of bitmasks over m edges
    biclique_edge_counts = []

    for L_list, R_list in maximal_bicliques:
        mask = 0
        count = 0
        for u in L_list:
            for v in R_list:
                key = (min(u,v), max(u,v))
                if key in edge_idx:
                    idx = edge_idx[key]
                    if (mask >> idx) & 1 == 0:
                        mask |= (1 << idx)
                        count += 1
        biclique_edge_masks.append(mask)
        biclique_edge_counts.append(count)

    # Remove bicliques that are subsets of other bicliques (dominated)
    # A biclique i is dominated by j if edges(i) ⊆ edges(j)
    # But we need to be careful: the biclique must still be valid (all cross pairs are edges)
    # Since all our candidates are valid bicliques, subset relationship on edge sets is safe

    # Sort by size descending for efficiency
    order = sorted(range(num_candidates), key=lambda i: -biclique_edge_counts[i])
    
    # Filter out dominated bicliques
    keep = [True] * num_candidates
    for i in range(num_candidates):
        if not keep[i]:
            continue
        for j in range(i + 1, num_candidates):
            if not keep[j]:
                continue
            if biclique_edge_masks[j] & ~biclique_edge_masks[i] == 0:
                # j is subset of i, so j is dominated
                keep[j] = False

    filtered_indices = [i for i in range(num_candidates) if keep[i]]
    filtered_bicliques = [maximal_bicliques[i] for i in filtered_indices]
    filtered_masks = [biclique_edge_masks[i] for i in filtered_indices]
    filtered_counts = [biclique_edge_counts[i] for i in filtered_indices]

    num_filtered = len(filtered_bicliques)

    # Now solve the set cover problem using branch and bound
    # Lower bound: greedy fractional relaxation
    # Upper bound: best greedy solution found

    # Precompute for each edge, which bicliques cover it
    edge_to_bicliques = [[] for _ in range(m)]
    for i in range(num_filtered):
        mask = filtered_masks[i]
        mm = mask
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            edge_to_bicliques[idx].append(i)
            mm ^= lsb

    # Branch and bound
    best_solution = None
    best_solution_count = best_count  # upper bound from greedy

    # Use a priority queue or DFS with pruning
    # State: (remaining_edges_mask, current_selection, count)
    # Lower bound: sum over remaining edges of 1 / (number of bicliques that can still cover it)
    # But this is expensive to compute. Simpler lower bound: ceil(remaining_edges / max_biclique_size)

    # Let's use a simpler approach: iterative deepening or DFS with good ordering

    # Sort bicliques by coverage size descending
    bb_order = sorted(range(num_filtered), key=lambda i: -filtered_counts[i])

    # Precompute for each biclique in BB order, the edges it covers
    # and for each edge, the list of bicliques that cover it (in BB order)

    # For efficiency, use bitmasks
    # m can be up to ~130, so we need Python ints (arbitrary precision) which is fine

    # DFS branch and bound
    # remaining: bitmask of edges still to cover
    # selected: list of biclique indices selected so far
    # count: number of bicliques selected

    # Lower bound: for each remaining edge, we need at least 1 biclique.
    # Better: group remaining edges, and for each group, the minimum number of bicliques needed
    # Simple LB: ceil(num_remaining_edges / max_coverage)
    # where max_coverage is the maximum number of remaining edges any single biclique can cover

    # Even better: greedy lower bound
    def greedy_lb(remaining_mask):
        """Compute a lower bound using greedy set cover relaxation"""
        if remaining_mask == 0:
            return 0
        # Greedy: repeatedly pick the biclique that covers the most remaining edges
        # This gives an upper bound on the number of bicliques, not a lower bound
        # For a lower bound, use: for each edge, at least 1 biclique. But edges can be covered together.
        # Simple LB: 1 if remaining_mask != 0
        # Better LB: count of edges that are only covered by one biclique among the remaining
        # Even better: LP relaxation via greedy
        
        # Let's use: the number of edges that are "critical" (only covered by few bicliques)
        # Simple approach: LB = ceil(popcount(remaining_mask) / max_edges_per_biclique)
        # But max_edges_per_biclique depends on remaining edges
        
        # Let's just use: 1 if any edges remain, plus a heuristic
        # Actually, for a proper LB, use: for each edge e in remaining, let c(e) = number of bicliques that cover e
        # Then LB >= sum over e of 1/c(e) (this is a valid fractional LB)
        
        lb = 0.0
        mm = remaining_mask
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            c = len(edge_to_bicliques[idx])
            if c > 0:
                lb += 1.0 / c
            else:
                return float('inf')  # infeasible
            mm ^= lsb
        return math.ceil(lb)

    # This LB is computed over ALL bicliques, not just those not yet selected.
    # For a tighter LB, we should only count bicliques not yet in the selection.
    # But that's expensive. Let's start with this.

    # DFS with memoization on remaining_mask
    memo = {}  # remaining_mask -> min additional bicliques needed

    def dfs(remaining_mask, count, selected):
        nonlocal best_solution, best_solution_count

        if time.time() > deadline - 1:
            return

        if remaining_mask == 0:
            if count < best_solution_count:
                best_solution_count = count
                best_solution = list(selected)
            return

        # Pruning
        if count >= best_solution_count:
            return

        # Check memo
        if remaining_mask in memo:
            if count + memo[remaining_mask] >= best_solution_count:
                return
        else:
            # Compute LB
            lb = greedy_lb(remaining_mask)
            if count + lb >= best_solution_count:
                memo[remaining_mask] = lb
                return

        # Find the edge with the fewest covering bicliques (most constrained)
        # This helps with branching
        mm = remaining_mask
        best_edge = -1
        best_edge_count = float('inf')
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            c = len(edge_to_bicliques[idx])
            if c < best_edge_count:
                best_edge_count = c
                best_edge = idx
            mm ^= lsb

        if best_edge == -1:
            return

        # Branch on which biclique to use for best_edge
        candidates = edge_to_bicliques[best_edge]
        # Sort by coverage size descending
        candidates.sort(key=lambda i: -filtered_counts[i])

        for i in candidates:
            if time.time() > deadline - 1:
                return
            new_remaining = remaining_mask & ~filtered_masks[i]
            selected.append(i)
            dfs(new_remaining, count + 1, selected)
            selected.pop()
            if best_solution_count == 1:
                return

    # Run DFS
    dfs((1 << m) - 1, 0, [])

    # If B&B found a better solution, use it
    if best_solution is not None and best_solution_count < best_count:
        best_count = best_solution_count
        # Reconstruct the bicliques
        best_bicliques = [filtered_bicliques[i] for i in best_solution]
    else:
        # Use the greedy solution
        pass

    # Phase 3: Local search - try to remove redundant bicliques and merge
    def try_merge_and_improve():
        nonlocal best_bicliques, best_count

        if best_count <= 1:
            return

        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = [b[:] for b in best_bicliques]
            current_count = len(current)

            # Try removing redundant bicliques
            for i in range(len(current)):
                if time.time() > deadline:
                    return

                L_i = current[i][0]
                R_i = current[i][1]

                cross_edges_i = set()
                for u in L_i:
                    for v in R_i:
                        if u < v:
                            cross_edges_i.add((u, v))
                        else:
                            cross_edges_i.add((v, u))

                all_covered = True
                for (u, v) in cross_edges_i:
                    covered = False
                    for j in range(len(current)):
                        if j == i:
                            continue
                        L_j = current[j][0]
                        R_j = current[j][1]
                        if (u in L_j and v in R_j) or (u in R_j and v in L_j):
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break

                if all_covered:
                    current.pop(i)
                    improved = True
                    break

            if improved:
                best_bicliques = current
                best_count = len(current)
                continue

            # Try merging two bicliques
            for i in range(len(current)):
                if time.time() > deadline:
                    return
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        return

                    L_i, R_i = current[i]
                    L_j, R_j = current[j]

                    L_new = set(L_i) | set(L_j)
                    R_new = set(R_i) | set(R_j)

                    # Check if L_new × R_new is a valid biclique
                    valid = True
                    for u in L_new:
                        for v in R_new:
                            if u == v:
                                continue
                            if (u, v) not in edge_set and (v, u) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break

                    if not valid:
                        continue

                    # Check coverage
                    cross_i = set()
                    for u in L_i:
                        for v in R_i:
                            if u < v:
                                cross_i.add((u, v))
                            else:
                                cross_i.add((v, u))

                    cross_j = set()
                    for u in L_j:
                        for v in R_j:
                            if u < v:
                                cross_j.add((u, v))
                            else:
                                cross_j.add((v, u))

                    cross_new = set()
                    for u in L_new:
                        for v in R_new:
                            if u < v:
                                cross_new.add((u, v))
                            else:
                                cross_new.add((v, u))

                    if cross_new == cross_i | cross_j:
                        new_current = current[:i] + current[i+1:j] + current[j+1:]
                        new_current.append([list(L_new), list(R_new)])
                        if len(new_current) < best_count:
                            best_bicliques = new_current
                            best_count = len(new_current)
                            improved = True
                            break
                if improved:
                    break

    try_merge_and_improve()

    # Phase 4: More greedy restarts with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        b, c = try_greedy(restarts=10, seed=seed)
        if b is not None and c < best_count:
            best_bicliques = b
            best_count = c

    # Final: try one more round of merge/improve
    if time.time() < deadline:
        try_merge_and_improve()

    # Return best found
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])

    return result
