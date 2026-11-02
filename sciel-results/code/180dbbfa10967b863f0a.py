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

    # Precompute conflict bitmask for each node for fast intersection checks
    # Use a list of frozensets for O(1) membership and intersection
    conflict_sets = [frozenset(neighbors[i]) for i in range(N4)]

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

    def local_search_fast(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Optimized SA with O(1) not_in_set updates, swap moves, and position tracking."""
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

        # Maintain in_set as a set for O(1) membership and iteration
        in_set_set = set(in_set_indices.tolist())

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

            # Choose move type: add, swap, or remove
            r = rng.random()
            if r < 0.55:
                move_type = 'add'
            elif r < 0.85 and len(in_set_set) > 0:
                move_type = 'swap'
            else:
                move_type = 'remove'

            if move_type == 'add':
                # Sample a subset if not_in_set is large
                if len(not_in_set) > 300:
                    sample_size = min(300, len(not_in_set))
                    sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                    sample = np.array([not_in_set[i] for i in sample_indices])
                else:
                    sample = np.array(not_in_set)

                counts = count_conflicts[sample]
                min_count = int(np.min(counts))

                if min_count == 0:
                    zero_mask = counts == 0
                    if np.any(zero_mask):
                        zero_cands = sample[zero_mask]
                        v = int(zero_cands[rng.integers(len(zero_cands))])
                    else:
                        v = int(not_in_set[rng.integers(len(not_in_set))])
                else:
                    one_mask = counts == 1
                    if np.any(one_mask) and rng.random() < 0.7:
                        one_cands = sample[one_mask]
                        v = int(one_cands[rng.integers(len(one_cands))])
                    else:
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

                if accept:
                    # Find conflicts with in_set using set intersection
                    v_conflicts_with_inset = conflict_sets[v] & in_set_set
                    to_remove = list(v_conflicts_with_inset)
                    if len(to_remove) > 0:
                        # Vectorized count_conflicts update
                        to_remove_arr = np.array(to_remove, dtype=np.int64)
                        # For each removed node u, decrement count_conflicts of its neighbors
                        for u in to_remove:
                            in_set[u] = False
                            in_set_set.discard(u)
                            add_to_not_in_set(u)
                        # Vectorized: build a sparse update
                        # count_conflicts[neighbors[u]] -= 1 for each u in to_remove
                        # We can do this by accumulating
                        if len(to_remove) > 1:
                            # Use a temporary array for accumulation
                            update_mask = np.zeros(N4, dtype=np.int32)
                            for u in to_remove:
                                update_mask[neighbors[u]] -= 1
                            count_conflicts += update_mask
                        else:
                            count_conflicts[neighbors[to_remove[0]]] -= 1

                    in_set[v] = True
                    in_set_set.add(v)
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

            elif move_type == 'swap':
                # Pick a random in-set node and a random not-in-set node
                if len(in_set_set) == 0 or len(not_in_set) == 0:
                    stuck_counter += 1
                    continue

                u_list = list(in_set_set)
                u = u_list[rng.integers(len(u_list))]
                if len(not_in_set) > 300:
                    sample_size = min(300, len(not_in_set))
                    sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                    sample = np.array([not_in_set[i] for i in sample_indices])
                else:
                    sample = np.array(not_in_set)

                v = int(sample[rng.integers(len(sample))])

                # Compute delta for swap: remove u, add v
                u_is_nbr_of_v = bool(conflicts[v, u])
                effective_conflicts_v = count_conflicts[v] - (1 if u_is_nbr_of_v else 0)

                if effective_conflicts_v == 0:
                    # Valid swap, delta = 0
                    accept = rng.random() < 0.5
                else:
                    # Try swap with removals: remove u and all of v's other in_set neighbors
                    v_conflicts_with_inset = conflict_sets[v] & in_set_set
                    to_remove = list(v_conflicts_with_inset)
                    n_remove = len(to_remove)
                    delta = 1 - n_remove
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = rng.random() < 0.3
                    else:
                        if delta < -20 * T:
                            accept = False
                        else:
                            accept = rng.random() < np.exp(delta / T)

                if accept:
                    v_conflicts_with_inset = conflict_sets[v] & in_set_set
                    to_remove = list(v_conflicts_with_inset)
                    if len(to_remove) > 0:
                        for w in to_remove:
                            in_set[w] = False
                            in_set_set.discard(w)
                            add_to_not_in_set(w)
                        # Vectorized count_conflicts update
                        if len(to_remove) > 1:
                            update_mask = np.zeros(N4, dtype=np.int32)
                            for w in to_remove:
                                update_mask[neighbors[w]] -= 1
                            count_conflicts += update_mask
                        else:
                            count_conflicts[neighbors[to_remove[0]]] -= 1

                    in_set[v] = True
                    in_set_set.add(v)
                    current_size += (1 - len(to_remove))
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

            else:
                # Remove a node from in_set (to allow re-exploration)
                if len(in_set_set) == 0:
                    stuck_counter += 1
                    continue

                u_list = list(in_set_set)
                u = u_list[rng.integers(len(u_list))]
                if rng.random() < 0.1:
                    in_set[u] = False
                    in_set_set.discard(u)
                    current_size -= 1
                    count_conflicts[neighbors[u]] -= 1
                    add_to_not_in_set(u)
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            if stuck_counter > 600:
                T = max(T, 2.5)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_polish(in_set_init, max_time=15.0):
        """High-precision polish: iterative greedy add + vectorized swap optimization."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        polish_deadline = time.time() + max_time

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        in_set_set = set(in_set_indices.tolist())
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1

        # Phase 1: Greedy add all zero-conflict nodes
        while time.time() < polish_deadline - 2:
            changed = False
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            zero_cands = not_in[count_conflicts[not_in] == 0]
            if len(zero_cands) > 0:
                for v in zero_cands:
                    in_set[v] = True
                    in_set_set.add(int(v))
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                changed = True
            if not changed:
                break

        # Phase 2: Vectorized swap optimization
        while time.time() < polish_deadline:
            improved = False
            in_set_list = np.array(list(in_set_set), dtype=np.int64) if in_set_set else np.array([], dtype=np.int64)
            not_in_set_list = np.where(~in_set)[0]

            if len(in_set_list) == 0 or len(not_in_set_list) == 0:
                break

            # For each not-in-set node with count_conflicts <= 1, find its blocker
            cc_not_in = count_conflicts[not_in_set_list]
            
            # Zero-conflict nodes: add directly
            zero_mask = cc_not_in == 0
            if np.any(zero_mask):
                zero_cands = not_in_set_list[zero_mask]
                for v in zero_cands:
                    in_set[v] = True
                    in_set_set.add(int(v))
                    current_size += 1
                    count_conflicts[neighbors[v]] += 1
                improved = True
                continue

            # One-conflict nodes: find their single blocker
            one_mask = cc_not_in == 1
            if np.any(one_mask):
                one_cands = not_in_set_list[one_mask]
                
                # For each candidate, find its blocker using set intersection
                for v in one_cands:
                    if time.time() >= polish_deadline:
                        break
                    v_int = int(v)
                    v_conflicts_with_inset = conflict_sets[v_int] & in_set_set
                    if len(v_conflicts_with_inset) == 1:
                        blocker = next(iter(v_conflicts_with_inset))
                        # Neutral swap: remove blocker, add v
                        in_set[blocker] = False
                        in_set_set.discard(blocker)
                        in_set[v_int] = True
                        in_set_set.add(v_int)
                        count_conflicts[neighbors[blocker]] -= 1
                        count_conflicts[neighbors[v_int]] += 1
                        improved = True
                        break
            
            # Try 2-for-1 swaps: find in_set nodes whose removal allows adding 2+ non-conflicting nodes
            if not improved:
                # Find nodes with cc == 1 and group by their blocker
                one_cands_all = not_in_set_list[count_conflicts[not_in_set_list] == 1]
                if len(one_cands_all) >= 2:
                    # Group candidates by their blocker
                    blocker_groups = {}
                    for v in one_cands_all:
                        v_int = int(v)
                        v_conflicts_with_inset = conflict_sets[v_int] & in_set_set
                        if len(v_conflicts_with_inset) == 1:
                            blocker = next(iter(v_conflicts_with_inset))
                            if blocker not in blocker_groups:
                                blocker_groups[blocker] = []
                            blocker_groups[blocker].append(v_int)
                    
                    # For each blocker with >= 2 candidates, check if any pair is non-conflicting
                    for blocker, cands in blocker_groups.items():
                        if time.time() >= polish_deadline:
                            break
                        if len(cands) < 2:
                            continue
                        
                        # Check pairs for non-conflict
                        for i in range(len(cands)):
                            for j in range(i + 1, len(cands)):
                                if not conflicts[cands[i], cands[j]]:
                                    # Valid 2-for-1 swap
                                    in_set[blocker] = False
                                    in_set_set.discard(blocker)
                                    in_set[cands[i]] = True
                                    in_set_set.add(cands[i])
                                    in_set[cands[j]] = True
                                    in_set_set.add(cands[j])
                                    current_size += 1
                                    count_conflicts[neighbors[blocker]] -= 1
                                    count_conflicts[neighbors[cands[i]]] += 1
                                    count_conflicts[neighbors[cands[j]]] += 1
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
            
            if not improved:
                break

        return in_set, current_size

    # Main Loop
    n_restarts = 0
    max_restarts = 200

    while time.time() < deadline - 25 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        remaining_time = deadline - 25 - time.time()
        if remaining_time < 2:
            break

        sa_time = min(remaining_time * 0.5, 14.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search_fast(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

    # Final high-precision polish pass
    remaining_polish = deadline - 5 - time.time()
    if remaining_polish > 3:
        polished, polished_size = exact_polish(best_in_set, max_time=remaining_polish)
        if polished_size > best_size:
            best_size = polished_size
            best_in_set = polished

    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
