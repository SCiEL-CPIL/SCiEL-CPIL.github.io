"""Kissing number in dimension 11 - Combined Local Search + Simulated Annealing."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates: norm squared 4
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

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init(seed=42):
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng_local.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_grouped(in_set, max_time=10.0, seed=42):
        rng_local = np.random.default_rng(seed)
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        best_size = current_size
        best_in_set = in_set.copy()
        
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = conflict_count[not_in_set]
                
            # 1. Add 0-conflict (try multiple per iteration)
            zero_mask = counts == 0
            if np.any(zero_mask):
                zero_idx = np.where(zero_mask)[0]
                # Add up to 5 zero-conflict vertices at once
                n_add = min(5, len(zero_idx))
                chosen = rng_local.choice(zero_idx, size=n_add, replace=False)
                for v_idx in chosen:
                    v = not_in_set[v_idx]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                stuck = 0
                continue

            # 2. 1-2 Swaps: find vertices with exactly 1 conflict
            one_mask = counts == 1
            if np.any(one_mask):
                one_idx = np.where(one_mask)[0]
                # Limit how many one-conflict vertices we process to save time
                max_process = min(len(one_idx), 200)
                process_idx = one_idx[:max_process] if len(one_idx) <= max_process else rng_local.choice(one_idx, size=max_process, replace=False)
                
                # Build groups: map conflicting vertex -> list of one-conflict vertices
                groups = defaultdict(list)
                for idx in process_idx:
                    v = not_in_set[idx]
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_in_set) == 1:
                        groups[int(nbrs_in_set[0])].append(v)
                
                swap_found = False
                for u, vs in groups.items():
                    if len(vs) < 2:
                        continue
                    if len(vs) <= 20:
                        for i in range(len(vs)):
                            for j in range(i+1, len(vs)):
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    in_set[u] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    conflict_count[neighbors[u]] -= 1
                                    conflict_count[neighbors[v1]] += 1
                                    conflict_count[neighbors[v2]] += 1
                                    current_size += 1
                                    improved = True
                                    swap_found = True
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_in_set = in_set.copy()
                                    break
                            if swap_found:
                                break
                    else:
                        for _ in range(20):
                            i = rng_local.integers(len(vs))
                            j = rng_local.integers(len(vs))
                            if i == j: continue
                            v1, v2 = vs[i], vs[j]
                            if not conflicts[v1, v2]:
                                in_set[u] = False
                                in_set[v1] = True
                                in_set[v2] = True
                                conflict_count[neighbors[u]] -= 1
                                conflict_count[neighbors[v1]] += 1
                                conflict_count[neighbors[v2]] += 1
                                current_size += 1
                                improved = True
                                swap_found = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_set = in_set.copy()
                                break
                    if swap_found:
                        break
                if swap_found:
                    stuck = 0
                    continue

            # 3. 1-3 Swaps
            if not improved and np.any(one_mask):
                one_idx = np.where(one_mask)[0]
                max_process = min(len(one_idx), 100)
                process_idx = one_idx[:max_process] if len(one_idx) <= max_process else rng_local.choice(one_idx, size=max_process, replace=False)
                
                groups = defaultdict(list)
                for idx in process_idx:
                    v = not_in_set[idx]
                    nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_in_set) == 1:
                        groups[int(nbrs_in_set[0])].append(v)
                
                swap_found = False
                for u, vs in groups.items():
                    if len(vs) < 3:
                        continue
                    chosen_vs = []
                    for v in vs:
                        if all(not conflicts[v, cv] for cv in chosen_vs):
                            chosen_vs.append(v)
                            if len(chosen_vs) == 3:
                                break
                    if len(chosen_vs) == 3:
                        in_set[u] = False
                        for v in chosen_vs:
                            in_set[v] = True
                        conflict_count[neighbors[u]] -= 1
                        for v in chosen_vs:
                            conflict_count[neighbors[v]] += 1
                        current_size += 2
                        improved = True
                        swap_found = True
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set.copy()
                        break
                if swap_found:
                    stuck = 0
                    continue

            # 4. 2-2 Swaps
            if not improved and stuck > 2:
                two_mask = counts == 2
                if np.any(two_mask):
                    two_idx = np.where(two_mask)[0]
                    max_process = min(len(two_idx), 100)
                    process_idx = two_idx[:max_process] if len(two_idx) <= max_process else rng_local.choice(two_idx, size=max_process, replace=False)
                    
                    pairs = defaultdict(list)
                    for idx in process_idx:
                        v = not_in_set[idx]
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 2:
                            key = (int(min(nbrs_in_set)), int(max(nbrs_in_set)))
                            pairs[key].append(v)
                    
                    swap_found = False
                    for (u1, u2), vs in pairs.items():
                        if len(vs) < 2:
                            continue
                        if len(vs) <= 20:
                            for i in range(len(vs)):
                                for j in range(i+1, len(vs)):
                                    v1, v2 = vs[i], vs[j]
                                    if not conflicts[v1, v2]:
                                        in_set[u1] = False
                                        in_set[u2] = False
                                        in_set[v1] = True
                                        in_set[v2] = True
                                        conflict_count[neighbors[u1]] -= 1
                                        conflict_count[neighbors[u2]] -= 1
                                        conflict_count[neighbors[v1]] += 1
                                        conflict_count[neighbors[v2]] += 1
                                        improved = True
                                        swap_found = True
                                        break
                                if swap_found:
                                    break
                        else:
                            for _ in range(20):
                                i = rng_local.integers(len(vs))
                                j = rng_local.integers(len(vs))
                                if i == j: continue
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    in_set[u1] = False
                                    in_set[u2] = False
                                    in_set[v1] = True
                                    in_set[v2] = True
                                    conflict_count[neighbors[u1]] -= 1
                                    conflict_count[neighbors[u2]] -= 1
                                    conflict_count[neighbors[v1]] += 1
                                    conflict_count[neighbors[v2]] += 1
                                    improved = True
                                    swap_found = True
                                    break
                        if swap_found:
                            break
                    if swap_found:
                        stuck = 0
                        continue

            # 5. 2-for-0 move: add two non-conflicting vertices with no removals
            # (already handled by zero-conflict adds above)

            if not improved:
                stuck += 1
                if stuck > 8:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        n_remove = rng_local.integers(1, 5)
                        to_remove = rng_local.choice(in_set_indices, size=min(n_remove, len(in_set_indices)), replace=False)
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap/2-for-1 moves."""
        rng_local = np.random.default_rng(seed)
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

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0:
                break

            move_type = rng_local.random()

            if move_type < 0.5:
                # ADD move: try to add a vertex, removing any conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng_local.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng_local.random() < 0.3
                else:
                    accept = rng_local.random() < np.exp(delta / T)

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
                v = in_set_indices[rng_local.integers(len(in_set_indices))]
                delta = -1
                accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) > 0:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                v_remove = in_set_indices[rng_local.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[
                    (count_conflicts[not_in_set] <= 1)
                ]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng_local.integers(len(v_add_candidates))]

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
                    accept = delta > 0 or (rng_local.random() < np.exp(delta / T))
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
                # 2-for-1 move: remove two in-set, add one not-in-set
                if len(in_set_indices) >= 2:
                    v_add = not_in_set[rng_local.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    if len(conflicts_with) == 2:
                        to_remove = conflicts_with
                        delta = -1
                        accept = rng_local.random() < np.exp(delta / T)
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
                    stuck_counter += 1

            # Cooling
            T = max(min_T, T * cooling)
            
            if stuck_counter > 50:
                T = 5.0
                stuck_counter = 0

        return local_best_in_set, local_best_size

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        # Run grouped local search first
        ls_time = min(remaining_time * 0.35, 12.0)
        ls_result, ls_size = local_search_grouped(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
        # Then run SA on the best result
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            sa_time = min(remaining_time * 0.4, 10.0)
            sa_result, sa_size = sa_local_search(best_in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    n_restarts = 0
    while time.time() < deadline - 4 and n_restarts < 50:
        n_restarts += 1
        in_set = greedy_init(seed=42 + n_restarts)
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 4 - time.time()
        if remaining_time < 2:
            break
            
        # Alternate between grouped LS and SA with tighter time budgets
        if n_restarts % 2 == 0:
            ls_time = min(remaining_time * 0.25, 6.0)
            ls_result, ls_size = local_search_grouped(in_set, max_time=ls_time)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()
        else:
            sa_time = min(remaining_time * 0.25, 6.0)
            sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result.copy()

    # Final polish: repeatedly add all zero-conflict vertices until no more can be added
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            not_in_set = np.where(~best_in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                changed = True
        
        # Second polish pass: try 1-2 swaps one more time
        if time.time() < deadline - 1:
            not_in_set = np.where(~best_in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                one_mask = counts == 1
                if np.any(one_mask):
                    one_idx = np.where(one_mask)[0]
                    groups = defaultdict(list)
                    for idx in one_idx[:100]:
                        v = not_in_set[idx]
                        nbrs_in_set = np.where(conflicts[v] & best_in_set)[0]
                        if len(nbrs_in_set) == 1:
                            groups[int(nbrs_in_set[0])].append(v)
                    
                    for u, vs in groups.items():
                        if len(vs) < 2:
                            continue
                        found = False
                        for i in range(min(len(vs), 15)):
                            for j in range(i+1, min(len(vs), 15)):
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    best_in_set[u] = False
                                    best_in_set[v1] = True
                                    best_in_set[v2] = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            # Rebuild conflict count
                            conflict_count = np.zeros(N, dtype=np.int32)
                            for i in np.where(best_in_set)[0]:
                                conflict_count[neighbors[i]] += 1
                            break
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
