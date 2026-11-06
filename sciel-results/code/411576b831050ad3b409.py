import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 90s wall-clock, 5s safety margin

    # ---- Precompute ALL APs (for final evaluation) ----
    # AP = (start, step, length>=1) within [0, N)
    # We need to evaluate max |sum| over ALL APs at the end.
    # For search, we use a reduced "critical" set.

    # Full AP list for final scoring
    all_aps = []  # list of index arrays
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) >= 1:
                all_aps.append(np.array(terms, dtype=np.int32))

    num_full_aps = len(all_aps)

    # Critical APs: length >= 3, and step in {1,2,3} OR length >= 5
    # This drastically reduces the constraint set while keeping the hardest ones.
    critical_aps = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            L = len(terms)
            if L >= 3 and (d <= 3 or L >= 5):
                critical_aps.append(np.array(terms, dtype=np.int32))

    num_crit = len(critical_aps)

    # Build incidence: for each critical AP, which positions it covers
    # crit_M: (num_crit x N) float64
    crit_M = np.zeros((num_crit, N), dtype=np.float64)
    pos_to_crit = [[] for _ in range(N)]
    for j, terms in enumerate(critical_aps):
        for i in terms:
            crit_M[j, i] = 1.0
            pos_to_crit[i].append(j)

    pos_to_crit_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_crit]

    # Full AP incidence for final scoring (compute on the fly or precompute)
    # We'll compute sums for all APs using a smarter approach:
    # For step d, start s: sum = sum of signs[s::d]
    # Precompute nothing; just evaluate directly.

    def full_hard_max(signs):
        """Compute max |AP sum| over ALL APs."""
        max_abs = 0
        for d in range(1, N + 1):
            for s in range(0, N - d + 1):
                # AP: s, s+d, s+2d, ...
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

    def crit_sums(signs):
        return crit_M @ signs

    # ============================================================
    # WALKSAT with dynamic clause weights
    # ============================================================

    def walksat(signs_init, time_limit, seed):
        """
        WalkSAT-style local search on the critical AP constraints.
        Each "clause" is: |sum_{i in AP} x_i| <= D, where D is the current target.
        We minimize the number of violated clauses, with dynamic weights.
        
        Actually, let's be more direct: minimize the max |AP sum| via
        a penalty function with dynamic weights on each AP.
        """
        rng = random.Random(seed)
        signs = signs_init.copy().astype(np.float64)
        sums = crit_M @ signs  # (num_crit,)
        
        # Current max
        cur_max = int(np.max(np.abs(sums)))
        
        # Dynamic weights: initially uniform, updated based on violation frequency
        weights = np.ones(num_crit, dtype=np.float64)
        
        # Penalty = sum over APs of w_j * max(0, |sum_j| - target)^2
        # We'll use target = current_max - 1 (try to push below)
        # Or simply minimize sum w_j * sum_j^2 with emphasis on large ones.
        
        # Strategy: minimize sum_j w_j * max(0, |sum_j| - D)^2
        # where D is a running target. Start D = cur_max, decrease when stuck.
        
        D = cur_max
        best_signs = signs.copy()
        best_max = cur_max
        
        # Precompute for speed: for each position k, which crit APs it belongs to
        # and the column of crit_M
        crit_cols = [crit_M[:, k] for k in range(N)]  # list of (num_crit,) arrays
        
        steps = 0
        no_improve = 0
        max_no_improve = 50000
        
        while time.time() < time_limit:
            steps += 1
            if no_improve > max_no_improve:
                # Reduce target D
                if D > 1:
                    D = max(1, D - 1)
                    no_improve = 0
                    # Reset weights
                    weights = np.ones(num_crit, dtype=np.float64)
                else:
                    break
            
            # Choose a move: single flip or pair flip
            if rng.random() < 0.7:
                # Single flip at random position
                k = rng.randint(0, N - 1)
                delta = -2.0 * signs[k]
                col = crit_cols[k]
                new_sums = sums + delta * col
                new_max = int(np.max(np.abs(new_sums)))
                
                # Accept if it doesn't increase the max, or with probability
                # based on how much it changes (simulated annealing style)
                if new_max <= cur_max:
                    # Accept
                    signs[k] = -signs[k]
                    sums += delta * col
                    cur_max = new_max
                    no_improve = 0
                    
                    # Update weights: increase weight of most violated APs
                    abs_sums = np.abs(sums)
                    violated = abs_sums > D
                    if np.any(violated):
                        weights[violated] *= 1.1
                        # Normalize occasionally
                        if weights.max() > 100:
                            weights = weights / weights.max()
                else:
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
                
                if new_max <= cur_max:
                    signs[k] = -signs[k]
                    signs[l] = -signs[l]
                    sums += dk * crit_cols[k] + dl * crit_cols[l]
                    cur_max = new_max
                    no_improve = 0
                else:
                    no_improve += 1
            
            if cur_max < best_max:
                best_max = cur_max
                best_signs = signs.copy()
                # Update D
                D = max(1, best_max - 1)
                no_improve = 0
                # Decay weights
                weights = weights * 0.95
                weights = np.maximum(weights, 0.1)
        
        return best_signs, best_max

    # ============================================================
    # Greedy initialization (fast)
    # ============================================================
    
    def greedy_init(seed):
        rng = random.Random(seed)
        signs = np.zeros(N, dtype=np.float64)
        sums = np.zeros(num_crit, dtype=np.float64)
        
        for pos in range(N):
            idxs = pos_to_crit_arr[pos]
            best_sign = 1.0
            best_val = float('inf')
            for s in [1.0, -1.0]:
                new_sums = sums[idxs] + s
                m = np.max(np.abs(new_sums)) if len(new_sums) > 0 else 0.0
                if m < best_val:
                    best_val = m
                    best_sign = s
            signs[pos] = best_sign
            sums[idxs] += best_sign
        
        return signs

    # ============================================================
    # Main search: multi-start WalkSAT + final polish
    # ============================================================
    
    best_signs = None
    best_full_max = N + 1
    
    # Phase 1: Greedy initialization + WalkSAT from multiple starts
    seeds = [1, 2, 3, 7, 13, 42, 100, 256, 512, 1024]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        signs_init = greedy_init(seed)
        crit_max = crit_hard_max(signs_init)
        
        # Run WalkSAT
        time_left = deadline - 10 - time.time()
        if time_left < 2:
            break
        # Allocate time per start
        per_start_time = min(5.0, time_left / max(1, len(seeds) - seeds.index(seed)))
        
        signs, crit_max = walksat(signs_init, time.time() + per_start_time, seed)
        
        # Evaluate on FULL AP set
        full_max = full_hard_max(signs)
        
        if full_max < best_full_max:
            best_full_max = full_max
            best_signs = signs.copy()
    
    # Phase 2: Final polish with single-flip on critical APs
    if best_signs is not None and time.time() < deadline - 3:
        signs = best_signs.copy().astype(np.float64)
        sums = crit_M @ signs
        cur_max = int(np.max(np.abs(sums)))
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                col = crit_M[:, k]
                new_sums = sums + delta * col
                new_max = int(np.max(np.abs(new_sums)))
                if new_max < cur_max:
                    signs[k] = -signs[k]
                    sums += delta * col
                    cur_max = new_max
                    improved = True
        
        full_max = full_hard_max(signs)
        if full_max < best_full_max:
            best_full_max = full_max
            best_signs = signs.copy()

    if best_signs is None:
        # Fallback: random
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]
    
    return [int(s) for s in best_signs]
