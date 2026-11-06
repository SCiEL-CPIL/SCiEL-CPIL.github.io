import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 84.0  # 90s wall-clock, 6s safety margin

    # Precompute critical APs: length >= 3 and (step <= 3 or length >= 5)
    critical_aps = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            L = len(terms)
            if L >= 3 and (d <= 3 or L >= 5):
                critical_aps.append(terms)

    num_crit = len(critical_aps)

    # Build incidence matrix and position-to-AP mapping
    crit_M = np.zeros((num_crit, N), dtype=np.float64)
    pos_to_crit = [[] for _ in range(N)]
    for j, terms in enumerate(critical_aps):
        for i in terms:
            crit_M[j, i] = 1.0
            pos_to_crit[i].append(j)

    pos_to_crit_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_crit]

    # Precompute columns for fast flip evaluation
    crit_cols = [crit_M[:, k].copy() for k in range(N)]

    def full_hard_max(signs):
        """Compute max |AP sum| over ALL APs."""
        max_abs = 0
        for d in range(1, N + 1):
            for s in range(0, N - d + 1):
                total = 0
                idx = s
                while idx < N:
                    total += signs[idx]
                    idx += d
                if total > max_abs:
                    max_abs = total
                elif -total > max_abs:
                    max_abs = -total
        return int(max_abs)

    def crit_hard_max(signs):
        """Compute max |AP sum| over critical APs only."""
        sums = crit_M @ signs
        return int(np.max(np.abs(sums)))

    def greedy_init(seed):
        rng = random.Random(seed)
        signs = np.zeros(N, dtype=np.float64)
        sums = np.zeros(num_crit, dtype=np.float64)
        
        for pos in range(N):
            idxs = pos_to_crit_arr[pos]
            best_sign = 1.0
            best_val = float('inf')
            for s in [1.0, -1.0]:
                if len(idxs) > 0:
                    new_sums = sums[idxs] + s
                    m = float(np.max(np.abs(new_sums)))
                else:
                    m = 0.0
                if m < best_val:
                    best_val = m
                    best_sign = s
            signs[pos] = best_sign
            if len(idxs) > 0:
                sums[idxs] += best_sign
        
        return signs

    def walksat(signs_init, time_limit, seed):
        """
        WalkSAT-style local search with simulated annealing on critical AP constraints.
        """
        rng = random.Random(seed)
        signs = signs_init.copy().astype(np.float64)
        sums = crit_M @ signs
        
        cur_max = int(np.max(np.abs(sums)))
        best_signs = signs.copy()
        best_max = cur_max
        
        # Temperature for simulated annealing
        temp = 2.0
        min_temp = 0.1
        cooling = 0.99999
        
        steps = 0
        no_improve = 0
        max_no_improve = 30000
        
        while time.time() < time_limit:
            steps += 1
            if no_improve > max_no_improve:
                temp = max(min_temp, temp * 0.5)
                no_improve = 0
                if temp <= min_temp * 1.01:
                    break
            
            # Choose a move: single flip or pair flip
            if rng.random() < 0.75:
                # Single flip at random position
                k = rng.randint(0, N - 1)
                delta = -2.0 * signs[k]
                col = crit_cols[k]
                new_sums = sums + delta * col
                new_max = int(np.max(np.abs(new_sums)))
                
                delta_max = new_max - cur_max
                if delta_max <= 0:
                    # Accept improving or equal moves
                    signs[k] = -signs[k]
                    sums += delta * col
                    cur_max = new_max
                    no_improve = 0
                    if cur_max < best_max:
                        best_max = cur_max
                        best_signs = signs.copy()
                        temp = max(min_temp, temp * 0.9)
                else:
                    # Accept worsening moves with probability exp(-delta_max/temp)
                    if rng.random() < np.exp(-delta_max / temp):
                        signs[k] = -signs[k]
                        sums += delta * col
                        cur_max = new_max
                    no_improve += 1
            else:
                # Pair flip
                k = rng.randint(0, N - 1)
                l = rng.randint(0, N - 1)
                if k == l:
                    continue
                dk = -2.0 * signs[k]
                dl = -2.0 * signs[l]
                new_sums = sums + dk * crit_cols[k] + dl * crit_cols[l]
                new_max = int(np.max(np.abs(new_sums)))
                
                delta_max = new_max - cur_max
                if delta_max <= 0:
                    signs[k] = -signs[k]
                    signs[l] = -signs[l]
                    sums += dk * crit_cols[k] + dl * crit_cols[l]
                    cur_max = new_max
                    no_improve = 0
                    if cur_max < best_max:
                        best_max = cur_max
                        best_signs = signs.copy()
                        temp = max(min_temp, temp * 0.9)
                else:
                    if rng.random() < np.exp(-delta_max / temp):
                        signs[k] = -signs[k]
                        signs[l] = -signs[l]
                        sums += dk * crit_cols[k] + dl * crit_cols[l]
                        cur_max = new_max
                    no_improve += 1
            
            temp *= cooling
        
        return best_signs, best_max

    def polish(signs_init, time_limit):
        """Final high-precision polish: iterative single-flip improvements."""
        signs = signs_init.copy().astype(np.float64)
        sums = crit_M @ signs
        cur_max = int(np.max(np.abs(sums)))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                col = crit_cols[k]
                new_sums = sums + delta * col
                new_max = int(np.max(np.abs(new_sums)))
                if new_max < cur_max:
                    signs[k] = -signs[k]
                    sums += delta * col
                    cur_max = new_max
                    improved = True
        
        return signs, cur_max

    # Main search
    best_signs = None
    best_full_max = N + 1

    seeds = [1, 2, 3, 7, 13, 42, 100, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    for idx, seed in enumerate(seeds):
        if time.time() > deadline - 15:
            break
        
        signs_init = greedy_init(seed)
        
        time_left = deadline - 15 - time.time()
        if time_left < 3:
            break
        per_start_time = min(4.0, time_left / max(1, len(seeds) - idx))
        
        signs, crit_max = walksat(signs_init, time.time() + per_start_time, seed)
        
        full_max = full_hard_max(signs)
        
        if full_max < best_full_max:
            best_full_max = full_max
            best_signs = signs.copy()
    
    # Final polish
    if best_signs is not None and time.time() < deadline - 5:
        signs, _ = polish(best_signs, deadline - 5)
        full_max = full_hard_max(signs)
        if full_max < best_full_max:
            best_full_max = full_max
            best_signs = signs.copy()

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
    
    return [int(s) for s in best_signs]
