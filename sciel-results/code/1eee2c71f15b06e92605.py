"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate candidate vectors
    # Weight 1: one entry ±2 -> norm^2 = 4
    # Weight 2: two entries ±1 -> norm^2 = 2
    # Weight 3: three entries ±1 -> norm^2 = 3
    # Weight 4: four entries ±1 -> norm^2 = 4

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)

    v3_candidates = []
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v3_candidates.append(v)

    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    all_candidates = v1_candidates + v2_candidates + v3_candidates + v4_candidates
    vectors_all = np.array(all_candidates, dtype=np.int64)
    N = len(vectors_all)

    norms_sq = np.sum(vectors_all ** 2, axis=1)
    dots = vectors_all @ vectors_all.T

    # Conflict if dot > (n1^2 + n2^2 - 4) / 2
    threshold = (norms_sq[:, None] + norms_sq[None, :] - 4) / 2.0
    conflicts = dots > threshold
    np.fill_diagonal(conflicts, False)

    # Build neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # Build conflict as set of sets for fast lookup
    conflict_sets = [set(n) for n in neighbors]

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_init(seed):
        """Greedy independent set with randomized min-degree."""
        r = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            # Randomized min-degree: pick from lowest 20% degree vertices
            min_deg = np.min(degrees[alive_indices])
            # Allow some randomness: pick from vertices with degree <= min_deg + 2
            cands = alive_indices[degrees[alive_indices] <= min_deg + 2]
            if len(cands) == 0:
                cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[r.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def greedy_weighted_init(seed):
        """Greedy with degree-weighted random selection."""
        r = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            # Weight inversely by degree
            d = degrees[alive_indices].astype(float)
            d[d < 1] = 1
            weights = 1.0 / d
            weights /= weights.sum()
            chosen_idx = r.choice(len(alive_indices), p=weights)
            chosen = alive_indices[chosen_idx]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def local_search_2swap(in_set_init, max_time=8.0, seed=42):
        """Local search with add, remove, and 2-swap moves."""
        r = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        local_best_size = current_size
        local_best_in_set = in_set.copy()

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            # Choose move type: 60% add, 20% remove, 20% swap
            move_type = r.random()
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if move_type < 0.5 and len(not_in_set) > 0:
                # Add move
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                cands = not_in_set[counts <= min_count + 2]
                v = cands[r.integers(len(cands))]

                k = count_conflicts[v]
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = r.random() < 0.4
                else:
                    accept = r.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1

            elif move_type < 0.7 and len(in_set_indices) > 0:
                # Remove move
                v = in_set_indices[r.integers(len(in_set_indices))]
                k = count_conflicts[v]
                # Removing v: we lose v but its neighbors gain freedom
                # Net effect: -1 + (number of neighbors that become conflict-free is complex)
                # Simplified: just remove, delta = -1
                delta = -1
                accept = r.random() < np.exp(delta / T) * 0.5

                if accept:
                    in_set[v] = False
                    current_size -= 1
                    count_conflicts[neighbors[v]] -= 1

            else:
                # 2-swap: remove one, try to add one
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[r.integers(len(in_set_indices))]
                    # Try to find a good v_add that conflicts with few in_set
                    counts = count_conflicts[not_in_set]
                    min_count = np.min(counts)
                    cands = not_in_set[counts <= min_count + 3]
                    v_add = cands[r.integers(len(cands))]

                    if v_add == v_remove:
                        continue

                    # Check if v_add conflicts with v_remove
                    if v_remove in conflict_sets[v_add]:
                        # They conflict, so we can't add v_add while v_remove is present
                        # But we're removing v_remove, so check if v_add conflicts with other in_set
                        k = count_conflicts[v_add]
                        # After removing v_remove, k decreases by 1 if they conflict
                        # They do conflict, so effective k = k - 1
                        effective_k = k - 1
                        delta = 1 - effective_k  # net: -1 (remove) + 1 - effective_k (add) = -effective_k
                        # Wait: delta = (1 - effective_k) + (-1) = -effective_k
                        delta = -effective_k

                        if delta > 0:
                            accept = True
                        elif delta == 0:
                            accept = r.random() < 0.3
                        else:
                            accept = r.random() < np.exp(delta / T)

                        if accept:
                            # Remove v_remove
                            in_set[v_remove] = False
                            current_size -= 1
                            count_conflicts[neighbors[v_remove]] -= 1
                            # Add v_add
                            to_remove = np.where(conflicts[v_add] & in_set)[0]
                            if len(to_remove) > 0:
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors[u]] -= 1
                                current_size -= len(to_remove)
                            in_set[v_add] = True
                            current_size += 1
                            count_conflicts[neighbors[v_add]] += 1

                    else:
                        # v_add doesn't conflict with v_remove
                        k = count_conflicts[v_add]
                        delta = (1 - k) + (-1)  # add v_add, remove v_remove
                        delta = -k

                        if delta > 0:
                            accept = True
                        elif delta == 0:
                            accept = r.random() < 0.3
                        else:
                            accept = r.random() < np.exp(delta / T)

                        if accept:
                            in_set[v_remove] = False
                            current_size -= 1
                            count_conflicts[neighbors[v_remove]] -= 1
                            to_remove = np.where(conflicts[v_add] & in_set)[0]
                            if len(to_remove) > 0:
                                in_set[to_remove] = False
                                for u in to_remove:
                                    count_conflicts[neighbors[u]] -= 1
                                current_size -= len(to_remove)
                            in_set[v_add] = True
                            current_size += 1
                            count_conflicts[neighbors[v_add]] += 1

            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1

            if stuck_counter > 1000:
                T = max(T, 2.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def local_polish(in_set_init, max_iters=20000):
        """Exact local search: try all improving single moves and 2-swaps."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1

            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            # Try all add moves
            if len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                # Find all vertices with minimum conflict count
                best_cands = not_in_set[counts == min_count]
                if min_count == 0:
                    # Can add for free
                    for v in best_cands:
                        in_set[v] = True
                        current_size += 1
                        count_conflicts[neighbors[v]] += 1
                        improved = True
                elif min_count == 1:
                    # Swap: remove one conflict, add this
                    # Try each candidate
                    for v in best_cands:
                        to_remove = np.where(conflicts[v] & in_set)[0]
                        if len(to_remove) == 1:
                            u = to_remove[0]
                            # Remove u, add v: net 0, but might unlock more
                            # For now, accept if it doesn't reduce size
                            in_set[u] = False
                            count_conflicts[neighbors[u]] -= 1
                            in_set[v] = True
                            count_conflicts[neighbors[v]] += 1
                            improved = True
                            break

            # Try 2-swaps: remove 1, add 1
            if not improved and len(in_set_indices) > 0 and len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                # Find v_add with min conflicts
                min_count = np.min(counts)
                best_adds = not_in_set[counts == min_count]

                for v_add in best_adds[:50]:  # Limit for speed
                    to_remove = np.where(conflicts[v_add] & in_set)[0]
                    if len(to_remove) == 1:
                        v_remove = to_remove[0]
                        # Check if this swap is beneficial
                        # After removing v_remove and adding v_add:
                        # v_add's conflicts with remaining in_set: 0 (we removed its only conflict)
                        # v_remove's neighbors: they lose one conflict
                        # Net: size unchanged, but might enable further additions
                        # Accept for now
                        in_set[v_remove] = False
                        count_conflicts[neighbors[v_remove]] -= 1
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_add]] += 1
                        improved = True
                        break

        return in_set, current_size

    # Multiple restarts with different initializations
    n_restarts = 0
    max_restarts = 200

    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        seed = 42 + n_restarts * 7

        # Choose initialization method
        if n_restarts % 3 == 0:
            in_set = greedy_init(seed)
        elif n_restarts % 3 == 1:
            in_set = greedy_weighted_init(seed)
        else:
            in_set = greedy_init(seed)

        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # SA local search
        remaining_time = deadline - 15 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.4, 8.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = local_search_2swap(in_set, max_time=sa_time, seed=seed)

        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()

        # Exact local polish
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 2:
            polish_result, polish_size = local_polish(sa_result, max_iters=10000)
            if polish_size > best_size:
                best_size = polish_size
                best_in_set = polish_result.copy()

    # Final polish on the best solution
    remaining_time = deadline - 2 - time.time()
    if remaining_time > 1:
        final_result, final_size = local_polish(best_in_set, max_iters=20000)
        if final_size > best_size:
            best_size = final_size
            best_in_set = final_result

    final_indices = np.where(best_in_set)[0]
    result_vectors = vectors_all[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
