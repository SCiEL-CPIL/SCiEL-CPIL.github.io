"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # --- W4 vectors (weight 4, entries in {-1,0,1}, norm^2 = 4) ---
    # --- W1 vectors (weight 1, entries in {-2,0,2}, norm^2 = 4) ---
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

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

    # --- Reduce to support graph ---
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        support_to_indices.setdefault(support, []).append(i)

    supports = list(support_to_indices.keys())
    n_supports = len(supports)

    support_conflicts = np.zeros((n_supports, n_supports), dtype=bool)
    sup_sets = [set(s) for s in supports]
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                support_conflicts[i, j] = True
                support_conflicts[j, i] = True

    neighbors_sup = [np.where(support_conflicts[i])[0] for i in range(n_supports)]
    neighbor_counts_sup = np.array([len(n) for n in neighbors_sup], dtype=np.int32)

    rng = np.random.default_rng(42)

    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)

    def greedy_min_degree_supports():
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def spectral_init_supports():
        """Use eigenvector of conflict graph adjacency to seed an independent set."""
        A = support_conflicts.astype(np.float64)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            # Largest eigenvalue corresponds to last column
            principal = eigenvectors[:, -1]
            # Select vertices with largest absolute components
            order = np.argsort(-np.abs(principal))
            in_set = np.zeros(n_supports, dtype=bool)
            for s in order:
                if not np.any(support_conflicts[s] & in_set):
                    in_set[s] = True
            return in_set
        except Exception:
            return greedy_min_degree_supports()

    def sa_local_search_supports(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 6.0
        cooling = 0.999
        min_T = 0.05

        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break

            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            min_cands = not_in_set[counts == min_count]
            s = min_cands[rng.integers(len(min_cands))]

            delta = 1 - min_count
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)

            if accept:
                to_remove = np.where(support_conflicts[s] & in_set)[0]
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors_sup[u]] -= 1

                in_set[s] = True
                current_size += delta
                count_conflicts[neighbors_sup[s]] += 1

                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1

            if stuck_counter > 1500:
                T = max(T, 3.0)
                stuck_counter = 0

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    def exact_local_polish(in_set_init, max_time=15.0):
        """Deterministic hill-climb: add free nodes, best single swap, and
        best 2-for-1 swap; stop at a local optimum."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors_sup[i]] += 1

        polish_deadline = time.time() + max_time
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False

            # Pass A: add any free node (delta = +1, no removals)
            free = np.where((~in_set) & (count_conflicts == 0))[0]
            for s in free:
                in_set[s] = True
                current_size += 1
                count_conflicts[neighbors_sup[s]] += 1
            if len(free) > 0:
                improved = True

            # Pass B: best single swap (remove neighbors of s, add s)
            best_delta = 0
            best_s = -1
            for s in np.where(~in_set)[0]:
                k = count_conflicts[s]
                if k == 0:
                    continue
                d = 1 - k
                if d > best_delta:
                    best_delta = d
                    best_s = s
            if best_s >= 0:
                to_remove = np.where(support_conflicts[best_s] & in_set)[0]
                for u in to_remove:
                    in_set[u] = False
                    count_conflicts[neighbors_sup[u]] -= 1
                in_set[best_s] = True
                current_size += best_delta
                count_conflicts[neighbors_sup[best_s]] += 1
                improved = True

            # Pass C: best 2-for-1 swap (remove two in-set nodes, add one)
            in_set_nodes = np.where(in_set)[0]
            if len(in_set_nodes) >= 2:
                best_d2 = 0
                best_s2 = -1
                for s in np.where(~in_set)[0]:
                    k = count_conflicts[s]
                    if k <= 2:
                        continue
                    d = 1 - k  # negative; we want the least negative (max)
                    if d > best_d2:
                        best_d2 = d
                        best_s2 = s
                if best_s2 >= 0:
                    rem = np.where(support_conflicts[best_s2] & in_set)[0]
                    if len(rem) >= 2:
                        d2 = 1 - len(rem)
                        if d2 >= 0:
                            for u in rem:
                                in_set[u] = False
                                count_conflicts[neighbors_sup[u]] -= 1
                            in_set[best_s2] = True
                            current_size += d2
                            count_conflicts[neighbors_sup[best_s2]] += 1
                            improved = True

        return in_set, current_size

    # --- Multiple restarts on the (much smaller) support graph ---
    n_restarts = 0
    max_restarts = 400

    # First restart uses spectral initialization
    if time.time() < deadline - 8:
        in_set = spectral_init_supports()
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 3:
            sa_time = min(remaining_time * 0.6, 12.0)
            if sa_time < 2:
                sa_time = 2.0
            sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_support_set = sa_result.copy()

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree_supports()
        current_size = int(in_set.sum())

        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()

        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break

        sa_time = min(remaining_time * 0.6, 12.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()

        # Polish the current best with an exact local solve
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 4:
            polish_time = min(remaining_time * 0.5, 12.0)
            if polish_time > 3:
                polish_result, polish_size = exact_local_polish(best_support_set, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_support_set = polish_result.copy()

    # --- Final deterministic polish on the best support set ---
    final_polish_time = max(0.0, deadline - 2 - time.time())
    if final_polish_time > 1:
        fp_result, fp_size = exact_local_polish(best_support_set, max_time=final_polish_time)
        if fp_size > best_size:
            best_size = fp_size
            best_support_set = fp_result.copy()

    # --- Lift support independent set to full W4 vectors ---
    selected_supports = np.where(best_support_set)[0]
    all_v4 = []
    for si in selected_supports:
        all_v4.extend(support_to_indices[supports[si]])
    if len(all_v4) > 0:
        result_v4 = vectors_v4[np.array(all_v4, dtype=np.int64)]
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
