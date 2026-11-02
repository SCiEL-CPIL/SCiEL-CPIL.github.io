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
            principal = eigenvectors[:, -1]
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

        T = 10.0
        cooling = 0.9995
        min_T = 0.01

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

            # Random move: pick a node to add
            s = not_in_set[rng.integers(len(not_in_set))]
            
            k = count_conflicts[s]
            delta = 1 - k
            
            if delta > 0:
                accept = True
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

            # Kick: if stuck, randomly flip some nodes
            if stuck_counter > 500:
                # Remove 2 random nodes, add 2 random nodes
                in_set_nodes = np.where(in_set)[0]
                if len(in_set_nodes) >= 2:
                    rem = rng.choice(in_set_nodes, size=2, replace=False)
                    for u in rem:
                        in_set[u] = False
                        count_conflicts[neighbors_sup[u]] -= 1
                        current_size -= 1
                
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) >= 2:
                    add = rng.choice(not_in_set, size=2, replace=False)
                    for u in add:
                        # Check if valid (might conflict, but we accept for kick)
                        in_set[u] = True
                        count_conflicts[neighbors_sup[u]] += 1
                        current_size += 1
                
                stuck_counter = 0
                T = max(T, 5.0)

            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # --- Multiple restarts on the support graph ---
    n_restarts = 0
    max_restarts = 1000

    # First restart uses spectral initialization
    if time.time() < deadline - 8:
        in_set = spectral_init_supports()
        current_size = int(in_set.sum())
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        remaining_time = deadline - 8 - time.time()
        if remaining_time > 3:
            sa_time = min(remaining_time * 0.7, 15.0)
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

        sa_time = min(remaining_time * 0.7, 15.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)

        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()

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
