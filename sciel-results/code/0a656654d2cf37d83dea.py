"""Kissing number in dimension 11 - combined design."""
import numpy as np
import itertools
import time

D = 11

def _max_independent_set_conflict(conflicts, neighbors, neighbor_counts, rng, max_time=8.0):
    """Combined greedy + SA for max independent set on conflict graph."""
    N = len(neighbor_counts)
    
    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < max_time - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            if len(cands) > 1:
                scores = np.array([np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0.0 for c in cands])
                chosen = cands[int(np.argmin(scores))]
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

    def polish(in_set):
        """Greedy add every zero-conflict vertex."""
        in_set = in_set.copy()
        cc = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        changed = True
        while changed:
            changed = False
            cand = np.where((cc == 0) & (~in_set))[0]
            for v in cand:
                if cc[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    changed = True
        return in_set, int(in_set.sum())

    def sa_local_search(in_set_init, max_time=8.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        cc = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0

        while time.time() < sa_deadline and T > min_T:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            r = rng.random()

            if r < 0.6 and len(not_in_set) > 0:
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.35
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            elif r < 0.8 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1

            else:
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                cc[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            cc[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()

            T *= cooling

        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    # Main loop
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    n_restarts = 0
    max_restarts = 40
    start_time = time.time()
    deadline = start_time + max_time

    while time.time() < deadline - 12 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        if time.time() > deadline - 12:
            break

        in_set, cur_size = sa_local_search(in_set)
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        if n_restarts > 1 and time.time() < deadline - 15:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 4:
                to_remove = rng.choice(indices, size=int(rng.integers(3, 5)), replace=False)
                perturbed[to_remove] = False
            elif len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed, psize = sa_local_search(perturbed)
            if psize > best_size:
                best_size = psize
                best_in_set = perturbed.copy()

    return best_in_set, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    rng = np.random.default_rng(42)

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # Conflict matrix: two vectors conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts_v4 = dots > 2
    np.fill_diagonal(conflicts_v4, False)

    neighbors_v4 = [np.where(conflicts_v4[i])[0] for i in range(N4)]
    neighbor_counts_v4 = np.array([len(n) for n in neighbors_v4], dtype=np.int32)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N1 = len(vectors_v1)

    # Build combined conflict graph: W4 + W1
    N_total = N4 + N1
    all_vectors = np.vstack([vectors_v4, vectors_v1])
    all_dots = all_vectors @ all_vectors.T
    conflicts_all = all_dots > 2
    np.fill_diagonal(conflicts_all, False)

    neighbors_all = [np.where(conflicts_all[i])[0] for i in range(N_total)]
    neighbor_counts_all = np.array([len(n) for n in neighbors_all], dtype=np.int32)

    # Run combined independent set search
    best_in_set, best_size = _max_independent_set_conflict(
        conflicts_all, neighbors_all, neighbor_counts_all, rng, max_time=100.0
    )

    result_vectors = all_vectors[best_in_set]
    return result_vectors
