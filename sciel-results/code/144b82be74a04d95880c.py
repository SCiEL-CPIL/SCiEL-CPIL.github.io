"""Kissing number in dimension 11 - Improved Local Search with Grouped Swaps."""
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

    def greedy_init():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_grouped(in_set, max_time=10.0):
        in_set = in_set.copy()
        current_size = in_set.sum()
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
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                
                # 1. Add 0-conflict
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    stuck = 0
                    continue

                # 2. 1-2 Swaps
                one_conf = not_in_set[counts == 1]
                if len(one_conf) > 0:
                    groups = defaultdict(list)
                    for v in one_conf:
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 1:
                            u = nbrs_in_set[0]
                            groups[u].append(v)
                    
                    swap_found = False
                    for u, vs in groups.items():
                        if len(vs) >= 2:
                            if len(vs) < 20:
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
                                    i = rng.integers(len(vs))
                                    j = rng.integers(len(vs))
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
                if not improved:
                    groups = defaultdict(list)
                    for v in one_conf:
                        nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                        if len(nbrs_in_set) == 1:
                            u = nbrs_in_set[0]
                            groups[u].append(v)
                    
                    swap_found = False
                    for u, vs in groups.items():
                        if len(vs) >= 3:
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
                            break
                    if swap_found:
                        stuck = 0
                        continue

                # 4. 2-2 Swaps
                if not improved and stuck > 2:
                    two_conf = not_in_set[counts == 2]
                    if len(two_conf) > 0:
                        pairs = defaultdict(list)
                        for v in two_conf:
                            nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                            if len(nbrs_in_set) == 2:
                                key = (min(nbrs_in_set), max(nbrs_in_set))
                                pairs[key].append(v)
                        
                        swap_found = False
                        for (u1, u2), vs in pairs.items():
                            if len(vs) >= 2:
                                if len(vs) < 20:
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
                                        i = rng.integers(len(vs))
                                        j = rng.integers(len(vs))
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

            if not improved:
                stuck += 1
                if stuck > 10:
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        n_remove = rng.integers(1, 4)
                        to_remove = rng.choice(in_set_indices, size=min(n_remove, len(in_set_indices)), replace=False)
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        stuck = 0

        return best_in_set, best_size

    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.4, 15.0)
        ls_result, ls_size = local_search_grouped(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 30:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.3, 8.0)
        ls_result, ls_size = local_search_grouped(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
