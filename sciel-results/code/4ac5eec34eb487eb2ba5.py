import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2, 0, 2}, norm^2=4)
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

    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as arrays for faster SA updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def greedy_max_degree():
        """Greedy independent set with max-degree heuristic (diversity)."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search_fast(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Optimized SA with O(1) not_in_set updates using position tracking."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        # Maintain not_in_set as a list with position tracking for O(1) removal
        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {}
        for idx, v in enumerate(not_in_set):
            pos_in_not_in_set[v] = idx

        def remove_from_not_in_set(v):
            """O(1) removal from not_in_set list."""
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]

        def add_to_not_in_set(v):
            """O(1) addition to not_in_set list."""
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)

        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break

            # Sample a subset if not_in_set is large to speed up min finding
            if len(not_in_set) > 300:
                sample_size = 300
                sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                sample = np.array([not_in_set[i] for i in sample_indices])
            else:
                sample = np.array(not_in_set)

            counts = count_conflicts[sample]
            min_count = int(np.min(counts))

            if min_count == 0:
                # Add a 0-conflict node
                zero_mask = counts == 0
                if np.any(zero_mask):
                    zero_cands = sample[zero_mask]
                    v = int(zero_cands[rng.integers(len(zero_cands))])
                else:
                    v = int(not_in_set[rng.integers(len(not_in_set))])

                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # Prioritize 1-conflict nodes: they only need to remove 1 vertex (net 0)
            # but can enable cascading additions. Use lower threshold for acceptance.
            one_mask = counts == 1
            if np.any(one_mask) and rng.random() < 0.7:
                one_cands = sample[one_mask]
                v = int(one_cands[rng.integers(len(one_cands))])
                k = 1
                delta = 1 - k  # delta = 0
                # Accept 1-conflict swaps with moderate probability to explore
                accept = rng.random() < min(1.0, 0.3 + T * 0.5)
            else:
                # SA step: pick node with minimum conflict count from sample
                min_mask = counts == min_count
                min_cands = sample[min_mask]
                v = int(min_cands[rng.integers(len(min_cands))])
                k = int(count_conflicts[v])
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    if delta < -20 * T:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(delta / T)

            k = int(count_conflicts[v])
            delta = 1 - k

            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                if delta < -20 * T:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)

            if accept:
                # Remove neighbors of v that are in set
                to_remove = np.where(conflicts[v] & in_set)[0]

                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                        add_to_not_in_set(int(u))

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 800:
                T = max(T, 2.5)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, max_time=5.0):
        """Exact local search: try all single additions, single swaps, and 2-swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Try adding all nodes with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                zero_cands = not_in_set[counts == 0]
                for v in zero_cands:
                    in_set[v] = True
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                    improved = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            # Try single swaps: for each node in set, temporarily remove and see if we can add more
            in_set_indices = np.where(in_set)[0].copy()
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break

                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                # Find all nodes that can now be added (0 conflicts)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) > 0:
                    counts = count_conflicts[not_in_set]
                    addable = not_in_set[counts == 0]
                    if len(addable) > 1:  # Need at least 2 to improve
                        for v in addable:
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                            improved = True
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                        else:
                            for v in addable:
                                in_set[v] = False
                                current_size -= 1
                                count_conflicts[neighbors[v]] -= 1
                            in_set[i] = True
                            current_size += 1
                            count_conflicts[neighbors[i]] += 1
                    elif len(addable) == 1:
                        v = addable[0]
                        if v != i:
                            in_set[v] = True
                            count_conflicts[neighbors[v]] += 1
                            improved = True
                        else:
                            in_set[i] = True
                            current_size += 1
                            count_conflicts[neighbors[i]] += 1
                    else:
                        in_set[i] = True
                        current_size += 1
                        count_conflicts[neighbors[i]] += 1
                else:
                    in_set[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1

            # Try 2-swap moves: remove two vertices, add all freed vertices
            if time.time() < polish_deadline and current_size >= 2:
                in_set_indices = np.where(in_set)[0].copy()
                n_in = len(in_set_indices)
                # Limit pairs to keep runtime bounded
                max_pairs = min(n_in * (n_in - 1) // 2, 500)
                pair_count = 0
                for pi in range(n_in):
                    if time.time() >= polish_deadline or pair_count >= max_pairs:
                        break
                    i = in_set_indices[pi]
                    for pj in range(pi + 1, n_in):
                        if time.time() >= polish_deadline or pair_count >= max_pairs:
                            break
                        pair_count += 1
                        j = in_set_indices[pj]

                        # Remove i and j
                        in_set[i] = False
                        in_set[j] = False
                        current_size -= 2
                        count_conflicts[neighbors[i]] -= 1
                        count_conflicts[neighbors[j]] -= 1

                        # Find all addable nodes
                        not_in_set = np.where(~in_set)[0]
                        if len(not_in_set) > 0:
                            counts = count_conflicts[not_in_set]
                            addable = not_in_set[counts == 0]
                            if len(addable) > 2:
                                for v in addable:
                                    in_set[v] = True
                                    current_size += 1
                                    count_conflicts[neighbors[v]] += 1
                                improved = True
                                if current_size > local_best_size:
                                    local_best_size = current_size
                                    local_best_in_set = in_set.copy()
                                else:
                                    for v in addable:
                                        in_set[v] = False
                                        current_size -= 1
                                        count_conflicts[neighbors[v]] -= 1
                                    in_set[i] = True
                                    in_set[j] = True
                                    current_size += 2
                                    count_conflicts[neighbors[i]] += 1
                                    count_conflicts[neighbors[j]] += 1
                            else:
                                in_set[i] = True
                                in_set[j] = True
                                current_size += 2
                                count_conflicts[neighbors[i]] += 1
                                count_conflicts[neighbors[j]] += 1
                        else:
                            in_set[i] = True
                            in_set[j] = True
                            current_size += 2
                            count_conflicts[neighbors[i]] += 1
                            count_conflicts[neighbors[j]] += 1

        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 100

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # Alternate between greedy variants for diversity
        if n_restarts % 3 == 0:
            in_set = greedy_max_degree()
        else:
            in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.45, 14.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search_fast(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish pass on the SA result
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size >= current_size:
            polish_time = min(remaining_time * 0.35, 8.0)
            if polish_time > 1:
                polish_result, polish_size = local_search_fast(
                    sa_result, max_time=polish_time, min_T=0.02, cooling=0.9998
                )
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final exact polish on best solution (two passes for thoroughness)
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        final_polish_time = min(remaining_time * 0.85, 8.0)
        if final_polish_time > 0.5:
            final_result, final_size = exact_polish(best_in_set, max_time=final_polish_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result

            # Second polish pass on the improved solution
            remaining_time2 = deadline - 1 - time.time()
            if remaining_time2 > 1 and final_size > 0:
                second_polish_time = min(remaining_time2 * 0.7, 4.0)
                if second_polish_time > 0.5:
                    second_result, second_size = exact_polish(best_in_set, max_time=second_polish_time)
                    if second_size > best_size:
                        best_size = second_size
                        best_in_set = second_result

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
