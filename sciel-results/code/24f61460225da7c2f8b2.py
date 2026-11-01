"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def get_candidates(weights):
        candidates = []
        for w in weights:
            if w == 1:
                for pos in range(D):
                    for sign in [1, -1]:
                        v = [0] * D
                        v[pos] = 2 * sign
                        candidates.append(v)
            elif w == 2:
                for positions in itertools.combinations(range(D), 2):
                    for signs in itertools.product([1, -1], repeat=2):
                        v = [0] * D
                        for pos, s in zip(positions, signs):
                            v[pos] = s
                        candidates.append(v)
            elif w == 3:
                for positions in itertools.combinations(range(D), 3):
                    for signs in itertools.product([1, -1], repeat=3):
                        v = [0] * D
                        for pos, s in zip(positions, signs):
                            v[pos] = s
                        candidates.append(v)
            elif w == 4:
                for positions in itertools.combinations(range(D), 4):
                    for signs in itertools.product([1, -1], repeat=4):
                        v = [0] * D
                        for pos, s in zip(positions, signs):
                            v[pos] = s
                        candidates.append(v)
        return np.array(candidates, dtype=np.int64)

    def solve_for_r2(R2, candidates, max_time=30.0):
        if len(candidates) == 0:
            return np.array([], dtype=np.int64).reshape(0, D), 0
        
        N = len(candidates)
        norms_sq = np.sum(candidates ** 2, axis=1)
        dots = candidates @ candidates.T
        
        # Conflict if dist^2 < R2
        # dist^2 = n1^2 + n2^2 - 2*dot
        # n1^2 + n2^2 - 2*dot < R2
        # 2*dot > n1^2 + n2^2 - R2
        threshold = (norms_sq[:, None] + norms_sq[None, :] - R2) / 2.0
        conflicts = dots > threshold
        np.fill_diagonal(conflicts, False)
        
        neighbors = [np.where(conflicts[i])[0] for i in range(N)]
        neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
        
        rng = np.random.default_rng(42)
        
        best_size = 0
        best_in_set = np.zeros(N, dtype=bool)
        
        def greedy_init():
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
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

        def local_search(in_set_init, max_iters=2000):
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
                if len(not_in_set) == 0:
                    break
                
                best_gain = 0
                best_v = -1
                
                # Vectorized gain calculation
                gains = 1 - count_conflicts[not_in_set]
                max_gain = np.max(gains)
                
                if max_gain > 0:
                    best_idx = np.argmax(gains)
                    best_v = not_in_set[best_idx]
                    best_gain = max_gain
                    
                    to_remove = np.where(conflicts[best_v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[best_v] = True
                    current_size += best_gain
                    count_conflicts[neighbors[best_v]] += 1
                    improved = True
            
            return in_set, current_size

        n_restarts = 0
        max_restarts = 50
        local_deadline = start_time + max_time
        
        while time.time() < local_deadline and n_restarts < max_restarts:
            n_restarts += 1
            in_set = greedy_init()
            current_size = int(in_set.sum())
            
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
            
            ls_result, ls_size = local_search(in_set)
            if ls_size > best_size:
                best_size = ls_size
                best_in_set = ls_result.copy()
        
        final_indices = np.where(best_in_set)[0]
        return candidates[final_indices], best_size

    # Case 1: R2 = 2, Weight 2 only
    cands_w2 = get_candidates([2])
    res1, size1 = solve_for_r2(2, cands_w2, max_time=40.0)
    
    # Case 2: R2 = 3, Weight 2, 3
    if time.time() < deadline - 40:
        cands_w23 = get_candidates([2, 3])
        res2, size2 = solve_for_r2(3, cands_w23, max_time=40.0)
        if size2 > size1:
            res1, size1 = res2, size2
    else:
        size2 = 0

    # Case 3: R2 = 4, Weight 1, 2, 3, 4
    if time.time() < deadline - 40:
        cands_all = get_candidates([1, 2, 3, 4])
        res3, size3 = solve_for_r2(4, cands_all, max_time=45.0)
        if size3 > size1:
            res1, size1 = res3, size3

    return res1

# EVOLVE-BLOCK-END
