import time
import numpy as np
import random
from scipy.optimize import linprog

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0 # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as boolean masks
    # cov[c, w] is True if w is in ball of c
    cov = np.zeros((total, total), dtype=np.bool_)
    cov_indices = [None] * total  # store list of covered indices for each codeword

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        covered_words = [c]
        cov[c, c] = True

        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = True
                covered_words.append(w)

        cov_indices[c] = np.array(covered_words, dtype=np.int32)

    # Solve LP Relaxation to get weights
    # Minimize sum(x) subject to Ax >= 1, 0 <= x <= 1
    # A is cov matrix (cov[c, w] means c covers w)
    # We need A such that A @ x >= 1
    # Let A_lp[w, c] = cov[c, w]
    A_lp = cov.T # Shape (total, total)
    
    c_obj = np.ones(total)
    b_lb = np.ones(total)
    
    # bounds for x: 0 <= x <= 1
    bounds = [(0, 1)] * total
    
    try:
        res = linprog(c_obj, A_ub=None, b_ub=None, A_eq=None, b_eq=None, 
                      A_lb=A_lp, b_lb=b_lb, bounds=bounds, method='highs')
        if res.status == 0:
            lp_weights = res.x
        else:
            lp_weights = np.ones(total) / total # Fallback
    except Exception:
        lp_weights = np.ones(total) / total

    # Greedy construction with LP weights
    def greedy_build(seed=None, use_lp=True):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total
        active = np.ones(total, dtype=np.bool_)
        
        while uncovered > 0:
            # Compute gains: number of new words covered
            gains = cov @ (1 - covered)
            
            # If using LP, add a small bias to break ties or guide selection
            # We want to prefer codewords with high LP weight among those with max gain
            # Or just use LP weight to select if gains are equal?
            # Better: maximize gain + lambda * lp_weight
            # Since gain is integer and lp_weight is float, we can scale.
            # Let's just pick from top-k gainers randomly, weighted by LP?
            # Simpler: Find max gain. Among candidates with max gain, pick one with highest LP weight.
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            candidates = candidates[active]
            
            if len(candidates) > 1:
                if use_lp:
                    # Weighted random choice based on LP weights
                    weights = lp_weights[candidates]
                    weights = weights / np.sum(weights)
                    # To avoid numerical issues, add small epsilon
                    weights = weights + 1e-6
                    weights = weights / np.sum(weights)
                    best_c = np.random.choice(candidates, p=weights)
                else:
                    best_c = random.choice(candidates)
            else:
                best_c = candidates[0]
            
            code.append(best_c)
            active[best_c] = False
            covered[cov[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search with 1-for-1 swaps and 2-for-1 reductions
    def local_search(code_list, time_budget=5.0):
        in_code = np.zeros(total, dtype=np.bool_)
        for c in code_list:
            in_code[c] = True

        cnt = np.zeros(total, dtype=np.int32)
        for c in code_list:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        current_code = list(code_list)

        def try_delete():
            nonlocal best_size, best_code, current_code
            changed = True
            while changed:
                changed = False
                to_remove = []
                for c in current_code:
                    if np.min(cnt[cov[c]]) >= 2:
                        to_remove.append(c)

                if to_remove:
                    for c in to_remove:
                        current_code.remove(c)
                        in_code[c] = False
                        cnt[cov[c]] -= 1
                        changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

        try_delete()

        no_success = 0
        max_streak = 500

        # Precompute coverage sets for faster intersection?
        # cov is already a boolean matrix.
        
        while time.time() - t_start < time_budget:
            if len(current_code) == 0:
                break

            # 1. Try deletion
            deleted = False
            for c in current_code:
                if np.min(cnt[cov[c]]) >= 2:
                    current_code.remove(c)
                    in_code[c] = False
                    cnt[cov[c]] -= 1
                    deleted = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    break

            if deleted:
                continue

            # 2. Try 2-for-1 reduction (Remove 2, Add 1)
            # This is the key improvement
            found_2_for_1 = False
            if len(current_code) > 2:
                # Sample a few pairs to avoid O(K^2) full scan
                # We can randomly sample pairs or iterate if K is small
                # K is ~200, so K^2 is 40k. Checking each pair might be fast enough if optimized.
                # Let's try iterating over a random subset of pairs first.
                
                # To be safe on time, we'll try a limited number of random pairs
                pairs_checked = 0
                max_pairs = 100 # Adjust this based on time
                
                while pairs_checked < max_pairs and not found_2_for_1:
                    if time.time() - t_start > time_budget - 0.5:
                        break
                    
                    # Pick two distinct codewords
                    c1, c2 = random.sample(current_code, 2)
                    pairs_checked += 1
                    
                    # Find words only covered by c1 or c2
                    # cnt[w] == 1 and (cov[c1, w] or cov[c2, w])
                    # Actually, if cnt[w] == 1, it's covered by exactly one codeword in current_code.
                    # If w is in B(c1) and cnt[w]==1, it must be c1.
                    # If w is in B(c2) and cnt[w]==1, it must be c2.
                    # So U = {w | cnt[w]==1 and (cov[c1,w] or cov[c2,w])}
                    
                    mask_c1 = cov[c1]
                    mask_c2 = cov[c2]
                    mask_union = mask_c1 | mask_c2
                    critical_mask = (cnt == 1) & mask_union
                    critical_words = np.where(critical_mask)[0]
                    
                    if len(critical_words) == 0:
                        # We can remove both for free?
                        # If no critical words, then all words in B(c1) U B(c2) are covered by >=2 codewords.
                        # So we can remove both.
                        current_code.remove(c1)
                        in_code[c1] = False
                        cnt[cov[c1]] -= 1
                        current_code.remove(c2)
                        in_code[c2] = False
                        cnt[cov[c2]] -= 1
                        found_2_for_1 = True
                        try_delete()
                        break
                    
                    # Find a codeword c_new not in code that covers all critical_words
                    # Intersection of S_w for w in critical_words
                    # S_w = np.where(cov[:, w])[0]
                    # We need c_new in intersection.
                    
                    # Start with all non-code words
                    candidates = ~in_code
                    # Intersect with coverage of each critical word
                    for w in critical_words:
                        candidates &= cov[:, w]
                        if not np.any(candidates):
                            break
                    
                    if np.any(candidates):
                        c_new = np.random.choice(np.where(candidates)[0])
                        # Perform swap
                        current_code.remove(c1)
                        in_code[c1] = False
                        cnt[cov[c1]] -= 1
                        current_code.remove(c2)
                        in_code[c2] = False
                        cnt[cov[c2]] -= 1
                        
                        current_code.append(c_new)
                        in_code[c_new] = True
                        cnt[cov[c_new]] += 1
                        
                        found_2_for_1 = True
                        try_delete()
                        break

            if found_2_for_1:
                no_success = 0
                continue

            # 3. Try 1-for-1 swap
            c_remove = random.choice(current_code)
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                current_code.remove(c_remove)
                in_code[c_remove] = False
                cnt[cov[c_remove]] -= 1
                no_success = 0
                continue

            critical_words = np.where(critical_mask)[0]
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            candidates_mask &= ~in_code
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                c_add = random.choice(candidate_indices)
                current_code.remove(c_remove)
                in_code[c_remove] = False
                current_code.append(c_add)
                in_code[c_add] = True
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                no_success = 0
                try_delete()
            else:
                no_success += 1

            # 4. Perturb if stuck
            if no_success > max_streak:
                rand_c = random.randint(0, total - 1)
                if not in_code[rand_c]:
                    current_code.append(rand_c)
                    in_code[rand_c] = True
                    cnt[cov[rand_c]] += 1
                    try_delete()
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS with LP guidance
    greedy_code = greedy_build(seed=42, use_lp=True)
    best_code, best_size = local_search(greedy_code, time_budget=15.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 10.0 and best_size_overall > 186:
        seed += 1
        greedy_code = greedy_build(seed=seed, use_lp=True)
        time_left = deadline - 10.0 - time.time()
        tb = min(8.0, max(2.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish
    if time.time() < deadline - 3.0 and best_code_overall:
        time_left = deadline - 3.0 - time.time()
        if time_left > 1.0:
            best_code, best_size = local_search(best_code_overall, time_budget=time_left)
            if best_size < best_size_overall:
                best_size_overall = best_size
                best_code_overall = best_code

    # Convert to list of lists
    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
