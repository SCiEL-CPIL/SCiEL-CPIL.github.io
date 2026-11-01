import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Group W4 vectors by their support (tuple of nonzero positions)
    # For a given support, all 16 sign variants are mutually compatible
    # (dot product between any two distinct sign variants on same support is 0 or ±2, never > 2)
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)

    supports = list(support_to_indices.keys())
    n_supports = len(supports)

    # Build support-level conflict matrix:
    # Two supports conflict if they share >= 3 positions
    # (if they share 3 positions, there exist sign variants with dot >= 3 > 2)
    # If they share <= 2 positions, all sign variants are compatible (max |dot| <= 2)
    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True

    # Each support contributes 16 vectors if selected
    vectors_per_support = 16

    # W1 vectors are all compatible with all W4 vectors (dot = 0 or ±1, |dot| <= 2)
    # So we just add all W1 vectors to the result.

    # Now solve independent set on support graph
    # This is a much smaller problem: ~330 nodes vs 3328
    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)

    def greedy_min_degree_supports(seed_rng=None):
        """Greedy independent set with min-degree heuristic on support graph."""
        r = seed_rng if seed_rng is not None else rng
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[r.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def sa_local_search_supports(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995, seed_rng=None):
        """Hybrid Greedy + Simulated Annealing with 1-add and 2-swap moves on support graph."""
        r = seed_rng if seed_rng is not None else rng
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors_sup[i]] += 1

        # Cache not-in-set indices
        not_in_set = np.where(~in_set)[0]

        step = 0
        while time.time() < sa_deadline:
            step += 1
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)

            if min_count == 0:
                # Add a 0-conflict node
                zero_cands = not_in_set[counts == 0]
                s = zero_cands[r.integers(len(zero_cands))]

                in_set[s] = True
                current_size += 1
                count_conflicts[neighbors_sup[s]] += 1

                # Remove s from not_in_set
                mask = not_in_set != s
                not_in_set = not_in_set[mask]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # Every 8 steps, try a 2-swap move for better exploration
            if step % 8 == 0 and current_size >= 2:
                # Pick two random nodes to remove from in_set
                in_set_idx = np.where(in_set)[0]
                if len(in_set_idx) >= 2:
                    i1, i2 = r.choice(in_set_idx, size=2, replace=False)

                    # Temporarily remove both
                    in_set[i1] = False
                    in_set[i2] = False
                    current_size -= 2
                    count_conflicts[neighbors_sup[i1]] -= 1
                    count_conflicts[neighbors_sup[i2]] -= 1

                    # Find addable nodes (not in set, 0 conflicts)
                    temp_not_in = np.concatenate([not_in_set, np.array([i1, i2])])
                    counts_now = count_conflicts[temp_not_in]
                    addable = temp_not_in[counts_now == 0]

                    if len(addable) >= 1:
                        # Try to add 1 or 2 nodes
                        if len(addable) >= 2 and r.random() < 0.3:
                            # Add 2 nodes
                            v1, v2 = r.choice(addable, size=2, replace=False)
                            in_set[v1] = True
                            in_set[v2] = True
                            current_size += 2
                            count_conflicts[neighbors_sup[v1]] += 1
                            count_conflicts[neighbors_sup[v2]] += 1
                            delta = 0
                        else:
                            # Add 1 node
                            v = addable[r.integers(len(addable))]
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors_sup[v]] += 1
                            delta = -1

                        # Update not_in_set
                        mask = np.isin(not_in_set, [i1, i2], invert=True)
                        not_in_set = not_in_set[mask]
                        # Add back i1, i2 if not already there
                        new_items = [x for x in [i1, i2] if not in_set[x]]
                        if new_items:
                            not_in_set = np.concatenate([not_in_set, np.array(new_items)])

                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        continue
                    else:
                        # No addable, restore
                        in_set[i1] = True
                        in_set[i2] = True
                        current_size += 2
                        count_conflicts[neighbors_sup[i1]] += 1
                        count_conflicts[neighbors_sup[i2]] += 1

                    continue

            # SA step: pick node with minimum conflict count
            min_cands = not_in_set[counts == min_count]
            s = min_cands[r.integers(len(min_cands))]

            k = min_count
            delta = 1 - k

            if delta > 0:
                accept = True
            elif delta == 0:
                accept = r.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = r.random() < np.exp(delta / T)

            if accept:
                # Remove neighbors of s that are in set
                to_remove = np.where(support_conflicts[s] & in_set)[0]

                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors_sup[u]] -= 1

                in_set[s] = True
                current_size += delta
                count_conflicts[neighbors_sup[s]] += 1

                # Update not_in_set: remove s, add back to_remove
                if len(to_remove) > 0:
                    mask = not_in_set != s
                    not_in_set = np.concatenate([not_in_set[mask], to_remove])
                else:
                    mask = not_in_set != s
                    not_in_set = not_in_set[mask]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish_supports(in_set_init, max_time=5.0):
        """Exact local search: try all single additions and 1-swap moves (optimized)."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors_sup[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding nodes with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                if len(zero_cands) > 0:
                    for s in zero_cands:
                        in_set[s] = True
                        current_size += 1
                        count_conflicts[neighbors_sup[s]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()

            # Try 1-swap: remove one node i, add node s if it becomes conflict-free
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors_sup[i]] -= 1

                # Only check nodes that conflict with i (they might now be addable)
                i_neighbors = neighbors_sup[i]
                # Among i's neighbors, find those not in set with 0 remaining conflicts
                cand_mask = ~in_set[i_neighbors]
                if np.any(cand_mask):
                    cand_indices = i_neighbors[cand_mask]
                    cand_counts = count_conflicts[cand_indices]
                    addable = cand_indices[cand_counts == 0]
                    if len(addable) > 0:
                        # Pick the one that unblocks the most other nodes
                        best_s = addable[0]
                        best_score = -1
                        for s in addable:
                            s_nbrs = neighbors_sup[s]
                            score = np.sum(~in_set[s_nbrs])
                            if score > best_score:
                                best_score = score
                                best_s = s
                        s = best_s
                        in_set[s] = True
                        current_size += 1
                        count_conflicts[neighbors_sup[s]] += 1
                        improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        # Keep i removed and s added (net zero change in size)
                        continue
                # No swap possible, restore i
                in_set[i] = True
                current_size += 1
                count_conflicts[neighbors_sup[i]] += 1

        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 200  # More restarts possible due to smaller problem

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Init with varying strategy
        if n_restarts % 3 == 0:
            # Standard min-degree
            in_set = greedy_min_degree_supports()
        elif n_restarts % 3 == 1:
            # Randomized greedy: pick from top-3 min-degree nodes
            in_set = np.zeros(n_supports, dtype=bool)
            alive = np.ones(n_supports, dtype=bool)
            degrees = neighbor_counts_sup.copy()
            while np.any(alive) and time.time() < deadline - 5:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                # Pick randomly from top candidates
                top_k = min(3, len(cands))
                chosen = cands[rng.integers(top_k)]
                in_set[chosen] = True
                nbrs = neighbors_sup[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
        else:
            # Random seed for diversity
            rng_local = np.random.default_rng(n_restarts * 12345)
            in_set = greedy_min_degree_supports(seed_rng=rng_local)

        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.45, 12.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()

        # 3. Polish with exact method
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.35, 6.0)
            if polish_time > 1:
                polish_result, polish_size = exact_polish_supports(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_support_set = polish_result.copy()

    # Final polish on best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        # Multiple exact polish passes
        for pass_num in range(5):
            remaining_time = deadline - 1 - pass_num * 0.5 - time.time()
            if remaining_time < 0.5:
                break
            polish_time = min(remaining_time * 0.5, 4.0)
            if polish_time < 0.5:
                break
            polish_result, polish_size = exact_polish_supports(best_support_set, max_time=polish_time)
            if polish_size > best_size:
                best_size = polish_size
                best_support_set = polish_result.copy()
            else:
                # No improvement, try SA with very low temperature
                remaining_time = deadline - 1 - time.time()
                if remaining_time > 1:
                    sa_time = min(remaining_time * 0.5, 2.0)
                    if sa_time > 0.5:
                        sa_result, sa_size = sa_local_search_supports(
                            best_support_set, max_time=sa_time, min_T=0.01, cooling=0.9999
                        )
                        if sa_size > best_size:
                            best_size = sa_size
                            best_support_set = sa_result.copy()
                        break

    # Expand support selection back to full vectors
    final_indices = []
    for s_idx in np.where(best_support_set)[0]:
        final_indices.extend(support_to_indices[supports[s_idx]])

    result_v4 = vectors_v4[np.array(final_indices)] if final_indices else np.empty((0, D), dtype=np.int64)
    final_vectors = np.vstack([result_v4, vectors_v1]) if len(result_v4) > 0 else vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
