"""D11 kissing number: CSP-guided construction + SA local search."""
import numpy as np
import itertools
import time

D = 11
UPPER_BOUND = 165  # Known kissing number upper bound for d=11

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all candidate vectors with norm^2 = 4
    # Type W4: weight 4, entries in {-1,0,1}, norm^2 = 4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Type W1: weight 1, entry ±2, norm^2 = 4
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix: conflict if dot > 2
    # Since all norms are 2 (norm^2=4), the threshold for "distance >= max norm"
    # means ||u - v|| >= 2, i.e., ||u||^2 + ||v||^2 - 2*u.v >= 4
    # 4 + 4 - 2*u.v >= 4 => u.v <= 2
    # So conflict if u.v > 2
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
                    for c in cands[:min(len(cands), 50)]  # Limit for speed
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

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap/2-for-1 moves."""
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
                # Find not-in-set vertices that don't conflict with in_set except possibly v_remove
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
                # 2-for-1 move: remove two in-set vertices, add one not-in-set vertex
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v1 = in_set_indices[rng.integers(len(in_set_indices))]
                    remaining = np.setdiff1d(in_set_indices, [v1])
                    v2 = remaining[rng.integers(len(remaining))]
                    
                    v_add_candidates = not_in_set[
                        (count_conflicts[not_in_set] <= 2)
                    ]
                    if len(v_add_candidates) == 0:
                        v_add_candidates = not_in_set
                    v_add = v_add_candidates[rng.integers(len(v_add_candidates))]

                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[~np.isin(conflicts_with, [v1, v2])]

                    if len(other_conflicts) == 0:
                        in_set[v1] = False
                        in_set[v2] = False
                        in_set[v_add] = True
                        count_conflicts[neighbors[v1]] -= 1
                        count_conflicts[neighbors[v2]] -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        current_size -= 1
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            # Adaptive perturbation when stuck
            if stuck_counter > 50:
                n_remove = min(3, len(in_set_indices))
                if n_remove > 0:
                    to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                    for u in to_remove:
                        in_set[u] = False
                        count_conflicts[neighbors[u]] -= 1
                    current_size -= n_remove
                    stuck_counter = 0

            # Cool down
            T = max(min_T, T * cooling)

        return local_best_in_set, local_best_size

    # CSP-guided initialization: prioritize vectors with lower conflict degree
    # and use the upper bound to guide search intensity
    def csp_guided_init(seed=42):
        """Initialize using CSP-inspired ordering: prefer low-degree vertices first."""
        rng = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        # Order by degree, with some randomization for diversity
        order = np.argsort(degrees + rng.random(N) * 0.1)

        for idx in order:
            if not alive[idx]:
                continue
            if time.time() >= deadline - 5:
                break
            in_set[idx] = True
            nbrs = neighbors[idx]
            alive[nbrs] = False
            alive[idx] = False

        return in_set

    # Multiple restarts with different strategies
    n_restarts = 0
    while time.time() < deadline - 10 and n_restarts < 200:
        n_restarts += 1

        # Alternate between greedy and CSP-guided initialization
        if n_restarts % 2 == 0:
            in_set = greedy_min_degree(seed=n_restarts)
        else:
            in_set = csp_guided_init(seed=n_restarts)

        cur_size = int(in_set.sum())
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        # If we've reached or exceeded the upper bound, stop early
        if best_size >= UPPER_BOUND:
            break

        rem_time = deadline - 10 - time.time()
        if rem_time < 1.0:
            break

        # Allocate more time to SA when close to upper bound
        if best_size >= UPPER_BOUND * 0.9:
            ls_time = min(rem_time * 0.6, 10.0)
        else:
            ls_time = min(rem_time * 0.5, 8.0)

        ls_res, ls_sz = sa_local_search(in_set, max_time=ls_time, seed=n_restarts)
        if ls_sz > best_size:
            best_size = ls_sz
            best_in_set = ls_res.copy()

            # If we've reached the upper bound, stop
            if best_size >= UPPER_BOUND:
                break

    return all_candidates[best_in_set]
