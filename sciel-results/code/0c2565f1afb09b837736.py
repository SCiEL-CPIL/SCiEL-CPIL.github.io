import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as boolean masks
    cov = np.zeros((total, total), dtype=np.bool_)
    cov_indices = [None] * total

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

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            # Compute gains vectorized
            gains = cov @ (1 - covered)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = rng.choice(candidates)
            else:
                best_c = candidates[0]

            code.append(best_c)
            covered[cov[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # LP-inspired initialization: use a heuristic based on coverage density
    def lp_inspired_init(seed=None, time_budget=5.0):
        """Create initial solution using LP-like weighting of coverage importance."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        t_start = time.time()
        
        # Start with a small random set
        initial_size = max(20, total // 50)
        initial_set = rng.sample(range(total), min(initial_size, total))
        
        # Iteratively refine: weight words by how many current codewords cover them
        # Words with low coverage get higher weight in the objective
        covered = np.zeros(total, dtype=np.uint8)
        for c in initial_set:
            covered[cov[c]] = 1
        
        code = list(initial_set)
        
        # Refine by adding high-value codewords
        while time.time() - t_start < time_budget:
            # Compute importance of each word: 1 if uncovered, 0 if covered
            # Weight uncovered words more heavily
            weights = (1 - covered).astype(np.float64)
            
            # Find codeword that maximizes weighted coverage
            gains = cov @ weights
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = rng.choice(candidates)
            else:
                best_c = candidates[0]
            
            if best_c not in code:
                code.append(best_c)
                covered[cov[best_c]] = 1
            
            # Occasionally try to remove redundant codewords
            if len(code) > 50 and rng.random() < 0.3:
                for c in code[:]:
                    cnt = np.sum(covered[cov[c]] == 1)
                    # Count how many words are ONLY covered by this codeword
                    only_covered = np.sum((covered[cov[c]] == 1) & (cov @ (1 - covered).astype(np.uint8) == 0))
                    # Simplified: check if removing it leaves all its words covered by others
                    temp_covered = covered.copy()
                    temp_covered[cov[c]] = 0
                    # Check if any word becomes uncovered
                    if np.all(temp_covered | cov[best_c]):  # This is wrong, let me fix
                        pass
                    # Better approach: check if all words in ball(c) are covered by other codewords
                    other_coverage = np.zeros(total, dtype=np.uint8)
                    for other in code:
                        if other != c:
                            other_coverage[cov[other]] += 1
                    if np.all(other_coverage[cov[c]] >= 1):
                        code.remove(c)
                        covered[cov[c]] -= 1
                        break
        
        return code

    # Local search with 1-for-1 swaps, 2-for-1 improvements, and periodic greedy reduction
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
        max_streak = 800
        greedy_reduction_interval = 200

        def greedy_reduction_pass():
            """Run a quick greedy reduction on the current code to find new deletions."""
            nonlocal best_size, best_code, current_code, in_code, cnt
            
            if not current_code:
                return False
            
            # Try removing a random codeword and seeing if we can cover gaps with 0 or 1 additions
            for attempt in range(20):
                c_remove = random.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]
                critical_words = np.where(critical_mask)[0]
                
                if len(critical_words) == 0:
                    # Can delete for free
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    cnt[cov[c_remove]] -= 1
                    try_delete()
                    return True
                
                # Find if there's a single codeword that covers all critical words
                candidates_mask = np.all(cov[:, critical_words], axis=1)
                candidates_mask &= ~in_code
                candidate_indices = np.where(candidates_mask)[0]
                
                if len(candidate_indices) > 0:
                    # 1-for-1 swap found
                    c_add = candidate_indices[0]
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    current_code.append(c_add)
                    in_code[c_add] = True
                    cnt[cov[c_remove]] -= 1
                    cnt[cov[c_add]] += 1
                    try_delete()
                    return True
            
            return False

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

            # 2. Try 1-for-1 swap
            c_remove = random.choice(current_code)
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                current_code.remove(c_remove)
                in_code[c_remove] = False
                cnt[cov[c_remove]] -= 1
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

            # 3. Try 2-for-1 improvement (remove 1, add 2)
            if no_success > 300 and time.time() - t_start < time_budget - 0.3 and current_code:
                c_remove = random.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]
                critical_words = np.where(critical_mask)[0]

                if len(critical_words) > 0:
                    found = False
                    any_critical_mask = np.any(cov[:, critical_words], axis=1)
                    any_critical_mask &= ~in_code
                    any_critical_indices = np.where(any_critical_mask)[0]
                    
                    if len(any_critical_indices) > 0:
                        sample_size = min(100, len(any_critical_indices))
                        sampled = random.sample(list(any_critical_indices), sample_size)
                        
                        for i in range(len(sampled)):
                            c1 = sampled[i]
                            c1_critical = cov[c1, critical_words]
                            remaining_critical = critical_words[~c1_critical]
                            
                            if len(remaining_critical) == 0:
                                continue
                            
                            c2_mask = np.all(cov[:, remaining_critical], axis=1)
                            c2_mask &= ~in_code
                            c2_mask[c1] = False
                            c2_indices = np.where(c2_mask)[0]
                            
                            if len(c2_indices) > 0:
                                c2 = c2_indices[0]
                                current_code.remove(c_remove)
                                in_code[c_remove] = False
                                current_code.append(c1)
                                in_code[c1] = True
                                current_code.append(c2)
                                in_code[c2] = True
                                cnt[cov[c_remove]] -= 1
                                cnt[cov[c1]] += 1
                                cnt[cov[c2]] += 1
                                no_success = 0
                                try_delete()
                                found = True
                                break
                    
                    if not found:
                        no_success += 1
                else:
                    no_success += 1

            # 4. Periodic greedy reduction pass
            if no_success > greedy_reduction_interval:
                if greedy_reduction_pass():
                    no_success = 0
                else:
                    no_success = 0

            # 5. Perturb if stuck
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

    # First: greedy + LS with generous time
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=20.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Second: LP-inspired initialization + LS
    if time.time() < deadline - 15.0:
        lp_code = lp_inspired_init(seed=123, time_budget=3.0)
        time_left = deadline - 15.0 - time.time()
        tb = min(10.0, max(3.0, time_left / 4.0))
        best_code, best_size = local_search(lp_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Multi-restart with different seeds
    seed = 0
    while time.time() < deadline - 15.0 and best_size_overall > 165:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 15.0 - time.time()
        tb = min(12.0, max(3.0, time_left / 6.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish: run local search on the best solution with remaining time
    if time.time() < deadline - 5.0 and best_code_overall:
        time_left = deadline - 5.0 - time.time()
        if time_left > 2.0:
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
