import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as boolean masks
    # cov[c, w] = True if c covers w
    cov = np.zeros((total, total), dtype=np.bool_)

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        cov[c, c] = True

        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = True

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

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
                best_c = random.choice(candidates)
            else:
                best_c = candidates[0]

            code.append(best_c)
            covered[cov[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search with 1-for-1 swaps and 2-for-1 improvements
    def local_search(code_list, time_budget=5.0):
        # Use in_code array for O(1) lookup and faster iteration
        in_code = np.zeros(total, dtype=np.bool_)
        for c in code_list:
            in_code[c] = True
        
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_list:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        
        # Maintain a list of current codewords for fast iteration and random choice
        # We need to keep it in sync with in_code
        current_code = list(code_list)

        def try_delete():
            nonlocal best_size, best_code, current_code
            changed = True
            while changed:
                changed = False
                # Iterate over a copy or indices to handle removal safely
                # Or just check all and mark for removal
                to_remove = []
                for c in current_code:
                    # Check if c can be deleted: all words in cov[c] have cnt >= 2
                    if np.min(cnt[cov[c]]) >= 2:
                        to_remove.append(c)
                
                if to_remove:
                    for c in to_remove:
                        current_code.remove(c) # O(N) removal, but N is small
                        in_code[c] = False
                        cnt[cov[c]] -= 1
                        changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

        try_delete()

        no_success = 0
        max_streak = 1000

        while time.time() - t_start < time_budget:
            if len(current_code) == 0:
                break

            # 1. Try deletion
            deleted = False
            # Find one deletable codeword
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
            if current_code:
                c_remove = random.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]

                if not np.any(critical_mask):
                    # Should have been caught by deletion, but just in case
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    cnt[cov[c_remove]] -= 1
                    continue

                critical_words = np.where(critical_mask)[0]

                # Find candidates that cover all critical words and are not in code
                # cov[:, critical_words] shape (total, K)
                # all along axis 1
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
            if no_success > 500 and time.time() - t_start < time_budget - 0.5 and current_code:
                # Try to remove one codeword and add two
                c_remove = random.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]
                critical_words = np.where(critical_mask)[0]

                if len(critical_words) > 0:
                    # Find two codewords that together cover all critical words
                    # Sample random pairs
                    found = False
                    for _ in range(50):
                        c1 = random.randint(0, total - 1)
                        c2 = random.randint(0, total - 1)
                        if in_code[c1] or in_code[c2] or c1 == c2:
                            continue
                        if c1 == c_remove or c2 == c_remove:
                            continue

                        # Check if c1 and c2 together cover all critical words
                        if np.all(cov[c1, critical_words] | cov[c2, critical_words]):
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

            if no_success > max_streak:
                # Perturb: add a random codeword
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

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=15.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 10.0 and best_size_overall > 170:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 10.0 - time.time()
        tb = min(10.0, max(2.0, time_left / 8.0))
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
