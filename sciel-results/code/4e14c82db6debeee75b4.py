"""Kissing number in dimension 11 - D11 lattice constructive + SA polish."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all shortest vectors of D11 lattice:
    # Type W1: weight 1, entry ±2 (norm^2 = 4)
    # Type W4: weight 4, entries in {-1,0,1} with EVEN number of -1's (norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            n_neg = sum(1 for s in signs if s == -1)
            if n_neg % 2 == 0:  # Even number of negatives for D11
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                v4_candidates.append(v)

    all_candidates = np.array(v1_candidates + v4_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix: conflict if dot > 2 (since all norms are 2, threshold is 4/2 = 2)
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    def greedy_min_degree(seed=42):
        """Greedy independent set with min-degree heuristic."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]

            if len(cands) > 1:
                # Tie-break: prefer vertex whose neighbors have lower average degree
                scores = np.array([
                    np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0
                    for c in cands
                ])
                chosen = cands[np.argmin(scores)]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def zero_conflict_polish(in_set, max_time=2.0):
        """Add all vertices with zero conflicts repeatedly."""
        in_set = in_set.copy()
        current_size = int(in_set.sum())

        # Count conflicts for each vertex
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        polish_deadline = time.time() + max_time

        while time.time() < polish_deadline:
            improved = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            zero_conf = not_in_set[conflict_count[not_in_set] == 0]
            while len(zero_conf) > 0 and time.time() < polish_deadline:
                v = zero_conf[0]
                in_set[v] = True
                current_size += 1
                conflict_count[neighbors[v]] += 1
                improved = True

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()

                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                zero_conf = not_in_set[conflict_count[not_in_set] == 0]

            if not improved:
                break

        return local_best_in_set, local_best_size

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap moves."""
        rng = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0

        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng.random()

            if move_type < 0.5 and len(not_in_set) > 0:
                # ADD move: try to add a vertex, removing any conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.3
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.7 and len(in_set_indices) > 0:
                # REMOVE move: remove a random vertex to escape local optima
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 1)
                ]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    # Clean swap
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
                    # Complex swap: remove v_remove and all other conflicts, add v_add
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1

                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1

            else:
                # 2-for-1 swap: remove 1, add 2
                if len(in_set_indices) > 0 and len(not_in_set) > 1:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add_candidates = not_in_set[
                        (count_conflicts[not_in_set] <= 1)
                    ]
                    if len(v_add_candidates) >= 2:
                        idx1 = rng.integers(len(v_add_candidates))
                        idx2 = rng.integers(len(v_add_candidates))
                        if idx1 != idx2:
                            v1 = v_add_candidates[idx1]
                            v2 = v_add_candidates[idx2]

                            v1_conf = np.where(conflicts[v1] & in_set)[0]
                            v2_conf = np.where(conflicts[v2] & in_set)[0]

                            # Check both only conflict with v_remove
                            if (len(v1_conf) <= 1 and (len(v1_conf) == 0 or v1_conf[0] == v_remove) and
                                len(v2_conf) <= 1 and (len(v2_conf) == 0 or v2_conf[0] == v_remove) and
                                not conflicts[v1, v2]):
                                to_remove = np.where((conflicts[v1] | conflicts[v2]) & in_set)[0]
                                to_remove = to_remove[to_remove != v1]
                                to_remove = to_remove[to_remove != v2]
                                if len(to_remove) == 1 and to_remove[0] == v_remove:
                                    in_set[v_remove] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    count_conflicts[neighbors[v_remove]] -= 1
                                    count_conflicts[neighbors[v1]] += 1
                                    count_conflicts[neighbors[v2]] += 1
                                    current_size += 1

                                    if current_size > local_best_size:
                                        local_best_size = current_size
                                        local_best_in_set = in_set.copy()
                                        stuck_counter = 0
                                    else:
                                        stuck_counter += 1
                                else:
                                    stuck_counter += 1
                            else:
                                stuck_counter += 1
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            if stuck_counter > 30:
                # Perturb: remove 2-4 random vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) >= 2:
                    n_remove = min(4, len(in_set_indices))
                    to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                    for u in to_remove:
                        in_set[u] = False
                        current_size -= 1
                        count_conflicts[neighbors[u]] -= 1
                    stuck_counter = 0
                else:
                    break

            T = max(min_T, T * cooling)

        # Final zero-conflict polish
        in_set, current_size = zero_conflict_polish(in_set, max_time=min(1.0, max_time * 0.1))
        if current_size > local_best_size:
            local_best_size = current_size
            local_best_in_set = in_set.copy()

        return local_best_in_set, local_best_size

    # Multiple restarts with different strategies
    n_restarts = 0
    max_restarts = 1000

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Use different random seeds for diversity
        seed = n_restarts * 7919 + 42
        in_set = greedy_min_degree(seed=seed)

        current_size = in_set.sum()

        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()

        # Zero-conflict polish first
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 1.0:
            break

        polished_in_set, polished_size = zero_conflict_polish(in_set, max_time=min(1.0, remaining_time * 0.1))
        if polished_size > best_size:
            best_size = polished_size
            best_in_set = polished_in_set.copy()

        # Local search with SA
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 1.0:
            break

        ls_time = min(remaining_time * 0.35, 4.0)
        if ls_time < 0.5:
            ls_time = 0.5

        ls_result, ls_size = sa_local_search(polished_in_set, max_time=ls_time, seed=seed)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

        if time.time() > deadline - 10:
            break

    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]

    return result_vectors

# EVOLVE-BLOCK-END
