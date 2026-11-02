"""Kissing number in dimension 11 - Improved combined design with faster SA."""
import numpy as np
import itertools
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    # 1. Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # 2. Build Conflict Graph
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for fast access
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # 3. Greedy Min-Degree Init
    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            if len(cands) > 1:
                # Tie-break: pick vertex with smallest sum of neighbor degrees
                scores = np.array([np.sum(neighbor_counts[neighbors[c]]) for c in cands])
                chosen = cands[int(np.argmin(scores))]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees: for each remaining vertex, subtract number of conflicts with removed
                # This is expensive, so we do it in batches
                # Actually, let's just recompute degrees for remaining vertices periodically
                # For now, subtract the count of neighbors that were removed
                for u in rem:
                    # Count how many of u's neighbors are in to_remove
                    # This is O(deg(u)) per vertex, which is too slow
                    # Instead, we'll just use a simpler approach: don't update degrees, just use alive mask
                    pass
                # Simpler: just recompute degrees for alive vertices when needed
                # Actually, the original code does this:
                # degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                # Let's keep that but it's O(|rem| * |to_remove|)
                # For large graphs this could be slow, but N4=5280 is manageable
                # Let's do it efficiently
                if len(to_remove) > 0:
                    degrees[rem] -= np.sum(conflicts[rem][:, to_remove], axis=1)

        return in_set

    # 4. Polish function
    def polish(in_set):
        """Greedy add every zero-conflict vertex repeatedly until fixed point."""
        in_set = in_set.copy()
        cc = np.zeros(N4, dtype=np.int32)
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

    # 5. SA Local Search - Optimized
    rng = np.random.default_rng(42)

    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0
        step_count = 0

        while time.time() < sa_deadline:
            step_count += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0:
                break

            r = rng.random()

            if r < 0.55 and len(not_in_set) > 0:
                # ADD: pick vertex with fewest in-set conflicts
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
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
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1

            elif r < 0.75 and len(in_set_indices) > 0:
                # REMOVE a random in-set vertex
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    stuck += 1
                else:
                    stuck += 1

            else:
                # SWAP: remove one in-set, add one not-in-set
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        # Clean swap
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                        stuck += 1
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
                                stuck = 0
                            else:
                                stuck += 1
                        else:
                            stuck += 1

            if stuck > 2000:
                T = max(T, 2.0)
                stuck = 0

            T *= cooling
            if T < min_T:
                T = min_T

        # Final polish
        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # Run SA from Greedy
    in_set_greedy = greedy_min_degree()
    if time.time() < deadline - 15:
        sa_time = min((deadline - 15 - time.time()) * 0.4, 12.0)
        set_g, size_g = sa_local_search(in_set_greedy, max_time=sa_time)
        if size_g > best_size:
            best_size = size_g
            best_in_set = set_g.copy()

    # Additional restarts from best with perturbation
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        # Try 2 more restarts with perturbation
        for restart in range(2):
            if time.time() >= deadline - 10:
                break
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            n_rem = max(2, int(len(indices) * 0.15))
            if len(indices) > n_rem:
                to_remove = rng.choice(indices, size=n_rem, replace=False)
                perturbed[to_remove] = False
            
            sa_time = min(remaining / 2, 10.0)
            set_p, size_p = sa_local_search(perturbed, max_time=sa_time)
            if size_p > best_size:
                best_size = size_p
                best_in_set = set_p.copy()

    # Final result: selected W4 vectors + all W1 (norm-4) vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    v1_candidates = []
    for pos in range(D):
        v = [0] * D
        v[pos] = 2
        v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
