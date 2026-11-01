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

    def local_search(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Hybrid Greedy + Simulated Annealing Local Search with cached not-in-set."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()

        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        # Cache not-in-set indices
        not_in_set = np.where(~in_set)[0]

        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)

            if min_count == 0:
                # Add a 0-conflict node
                zero_cands = not_in_set[counts == 0]
                v = zero_cands[rng.integers(len(zero_cands))]

                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1

                # Remove v from not_in_set
                mask = not_in_set != v
                not_in_set = not_in_set[mask]

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue

            # SA step: pick node with minimum conflict count
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]

            k = min_count
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

                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1

                # Update not_in_set: remove v, add back to_remove
                if len(to_remove) > 0:
                    mask = not_in_set != v
                    not_in_set = np.concatenate([not_in_set[mask], to_remove])
                else:
                    mask = not_in_set != v
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

    def exact_polish(in_set_init, max_time=5.0):
        """Exact local search: try all single additions and 1-swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
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

            # Try adding nodes with 0 conflicts
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

            # Try 1-swap: remove one node i, add node v if it becomes conflict-free
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                if time.time() >= polish_deadline:
                    break
                # Temporarily remove i
                in_set[i] = False
                current_size -= 1
                count_conflicts[neighbors[i]] -= 1

                # Find nodes that can now be added (0 conflicts)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) > 0:
                    counts = count_conflicts[not_in_set]
                    addable = not_in_set[counts == 0]
                    # Only consider addable nodes that were actually blocked by i
                    # (i.e., they conflict with i) for a true swap
                    if len(addable) > 0:
                        # Check which addable nodes conflict with i
                        blocked_by_i = conflicts[i, addable]
                        swap_cands = addable[blocked_by_i]
                        if len(swap_cands) > 0:
                            # Pick the swap candidate that unblocks the most other nodes
                            # For simplicity, just add one
                            v = swap_cands[0]
                            in_set[v] = True
                            current_size += 1
                            count_conflicts[neighbors[v]] += 1
                            improved = True
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                            # Restore i since we did a swap (net zero change)
                            # Actually, the swap means we removed i and added v, so we stay
                            # But we need to keep i removed and v added
                            continue
                        else:
                            # No swap possible, restore i
                            in_set[i] = True
                            current_size += 1
                            count_conflicts[neighbors[i]] += 1
                    else:
                        # No addable nodes, restore i
                        in_set[i] = True
                        current_size += 1
                        count_conflicts[neighbors[i]] += 1
                else:
                    in_set[i] = True
                    current_size += 1
                    count_conflicts[neighbors[i]] += 1

        return local_best_in_set, local_best_size

    # Main Loop
    n_restarts = 0
    max_restarts = 60

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        # 1. Greedy Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.5, 10.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # 3. Polish (Second pass on best with tighter parameters)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3 and sa_size > current_size:
            polish_time = min(remaining_time * 0.4, 6.0)
            if polish_time > 1:
                polish_result, polish_size = local_search(
                    sa_result, max_time=polish_time, min_T=0.02, cooling=0.9998
                )
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()

    # Final exact polish on best solution (multiple passes)
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1 and best_size > 0:
        final_polish_time = min(remaining_time * 0.8, 5.0)
        if final_polish_time > 0.5:
            final_result, final_size = exact_polish(best_in_set, max_time=final_polish_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_result

            # Second polish pass on the improved solution
            remaining_time = deadline - 1 - time.time()
            if remaining_time > 1 and final_size > best_size:
                pass2_time = min(remaining_time * 0.7, 3.0)
                if pass2_time > 0.5:
                    pass2_result, pass2_size = exact_polish(best_in_set, max_time=pass2_time)
                    if pass2_size > best_size:
                        best_size = pass2_size
                        best_in_set = pass2_result

            # One more SA pass with very low temperature on the best
            remaining_time = deadline - 0.5 - time.time()
            if remaining_time > 1:
                sa2_time = min(remaining_time * 0.6, 2.0)
                if sa2_time > 0.5:
                    sa2_result, sa2_size = local_search(
                        best_in_set, max_time=sa2_time, min_T=0.01, cooling=0.9999
                    )
                    if sa2_size > best_size:
                        best_size = sa2_size
                        best_in_set = sa2_result

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
