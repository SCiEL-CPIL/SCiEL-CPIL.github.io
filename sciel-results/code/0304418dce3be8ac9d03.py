import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin under 120s

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as numpy int32 arrays for fast indexing
    # We also build the dense coverage matrix
    # cov[c, w] = 1 if c covers w
    # This is 2187 x 2187 for q=3, n=7
    cov = np.zeros((total, total), dtype=np.int8)
    
    # Precompute digit expansions for all words to speed up ball generation
    # Or just generate balls on the fly for the matrix
    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        
        # The word itself
        cov[c, c] = 1
        
        # Neighbors
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = 1

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            # Find the codeword that covers the most uncovered points
            # gains[c] = sum(covered[w] == 0 for w in ball(c))
            # This is equivalent to (1 - covered) @ cov.T
            # But cov is (total, total), so cov.T is (total, total)
            # (1 - covered) is (total,)
            # Result is (total,)
            
            # Optimization: Use matrix multiplication
            # uncovered_mask = 1 - covered
            # gains = cov.T @ uncovered_mask  # Wait, cov[c, w] is 1 if c covers w
            # We want sum over w of cov[c, w] * (1 - covered[w])
            # This is cov @ (1 - covered)
            
            gains = cov @ (1 - covered)
            
            # Random tie-breaking
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = random.choice(candidates)
            else:
                best_c = candidates[0]
                
            code.append(best_c)
            covered[cov[best_c] == 1] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        # Coverage count for each word
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[cov[c] == 1] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    # Check if removing c leaves any word with count 0
                    # new_cnt = cnt - cov[c]
                    # We need new_cnt.min() >= 1
                    # Equivalent to: for all w where cov[c, w]==1, cnt[w] >= 2
                    # Or simply: np.min(cnt[cov[c]==1]) >= 2
                    if np.min(cnt[cov[c] == 1]) >= 2:
                        code_set.remove(c)
                        cnt[cov[c] == 1] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        no_success = 0
        max_streak = 500

        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # 1. Try deletion
            deleted = False
            for c in list(code_set):
                if np.min(cnt[cov[c] == 1]) >= 2:
                    code_set.remove(c)
                    cnt[cov[c] == 1] -= 1
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break # Restart loop after modification
            
            if deleted:
                continue

            # 2. Try swap
            # Pick a random codeword to remove
            c_remove = random.choice(list(code_set))
            
            # Find critical words: covered only by c_remove
            # critical[w] = 1 if cnt[w] == 1 and cov[c_remove, w] == 1
            critical_mask = (cnt == 1) & (cov[c_remove] == 1)
            
            if not np.any(critical_mask):
                # c_remove is redundant? Should have been caught by deletion.
                # Maybe numerical issue or logic gap. Force delete?
                code_set.remove(c_remove)
                cnt[cov[c_remove] == 1] -= 1
                continue

            # Find a replacement c_add such that cov[c_add, w] == 1 for all critical w
            # We need c_add not in code_set
            # And c_add covers all critical words
            
            # Efficient search:
            # 1. Get list of critical word indices
            critical_words = np.where(critical_mask)[0]
            
            # 2. Find all codewords that cover ALL critical words
            # For each codeword c, check if cov[c, critical_words] are all 1
            # This is a matrix-vector operation or logical AND
            # cov is (total, total). critical_words is (k,)
            # We want c such that np.all(cov[c, critical_words] == 1)
            
            # To speed this up, we can precompute or just vectorize
            # candidates_mask = np.all(cov[:, critical_words] == 1, axis=1)
            # This creates a (total, k) array which might be large if k is large.
            # k is at most ball_size = 15. So (2187, 15) is small.
            
            candidates_mask = np.all(cov[:, critical_words] == 1, axis=1)
            
            # Exclude codewords already in the set
            in_set_mask = np.zeros(total, dtype=bool)
            for c in code_set:
                in_set_mask[c] = True
            candidates_mask &= ~in_set_mask
            
            candidate_indices = np.where(candidates_mask)[0]
            
            if len(candidate_indices) > 0:
                # Pick a random candidate
                c_add = random.choice(candidate_indices)
                
                # Perform swap
                code_set.remove(c_remove)
                code_set.add(c_add)
                
                cnt[cov[c_remove] == 1] -= 1
                cnt[cov[c_add] == 1] += 1
                
                no_success = 0
                try_delete()
            else:
                no_success += 1
                
            if no_success > max_streak:
                # Perturb: add a random codeword
                rand_c = random.randint(0, total - 1)
                if rand_c not in code_set:
                    code_set.add(rand_c)
                    cnt[cov[rand_c] == 1] += 1
                    try_delete()
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=10.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 5.0 and best_size_overall > 175: # Target better than 175
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 5.0 - time.time()
        tb = min(8.0, max(1.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
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
