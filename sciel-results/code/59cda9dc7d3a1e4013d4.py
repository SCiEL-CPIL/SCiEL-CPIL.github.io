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
    balls = [None] * total
    for i in range(total):
        temp = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        ball = [i]
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                ball.append(i + (v - orig) * base[coord])
        balls[i] = np.array(ball, dtype=np.int32)

    # Greedy construction with numpy
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0 and time.time() < deadline - 5.0:
            # Find uncovered points and pick one at random
            uncov_idx = np.where(covered == 0)[0]
            if len(uncov_idx) == 0:
                break
            target = uncov_idx[random.randint(0, len(uncov_idx) - 1)]
            
            # Find all codewords that cover this target
            # A codeword c covers target if target is in balls[c]
            # We can precompute for each point which codewords cover it
            # But that's expensive. Instead, scan all codewords and count new coverage.
            
            best_c = -1
            max_new = -1
            
            # To speed up, we can first check a random subset
            # But for correctness, we need to check all. Let's use numpy vectorization.
            
            # For each candidate c, compute how many new points it covers
            # new_cov[c] = count of x in balls[c] where covered[x] == 0
            
            # This is expensive to do for all c at once. Let's do a two-pass:
            # First pass: random sample of candidates
            # Second pass: full scan if needed
            
            sample_size = min(200, total)
            candidates = random.sample(range(total), sample_size)
            
            for c in candidates:
                b = balls[c]
                new_cov = np.count_nonzero(covered[b] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            # If we found a good candidate, use it. Otherwise, do full scan.
            if max_new < ball_size // 2:
                for c in range(total):
                    if c in set(candidates):
                        continue
                    b = balls[c]
                    new_cov = np.count_nonzero(covered[b] == 0)
                    if new_cov > max_new:
                        max_new = new_cov
                        best_c = c
                        if max_new == ball_size:
                            break
            
            if best_c < 0:
                break
            code.append(best_c)
            covered[balls[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search with delete + swap
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[balls[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    b = balls[c]
                    if np.min(cnt[b]) >= 2:
                        code_set.remove(c)
                        cnt[b] -= 1
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

            # Alternate between 1-for-1, 2-for-1, and 1-for-2 swaps
            move_type = random.randint(0, 2)
            
            valid_move = False
            
            if move_type == 0:
                # 1-for-1 swap
                if len(code_set) < 2:
                    continue
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)

                if c_add in code_set or c_add == c_remove:
                    continue

                b_rem = balls[c_remove]
                b_add = balls[c_add]

                # Check validity using numpy
                # For all x in b_rem, if x not in b_add, need cnt[x] >= 2
                # Equivalently: for all x in b_rem, need cnt[x] - (1 if x in b_add else 0) >= 2
                # i.e., cnt[x] >= 2 + (1 if x in b_add else 0)
                
                # Create a mask for b_add
                in_add = np.zeros(total, dtype=np.uint8)
                in_add[b_add] = 1
                
                # For x in b_rem: need cnt[x] >= 2 + in_add[x]
                # Check: min over x in b_rem of (cnt[x] - in_add[x]) >= 2
                vals = cnt[b_rem] - in_add[b_rem]
                if np.min(vals) >= 2:
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[b_rem] -= 1
                    cnt[b_add] += 1
                    valid_move = True

            elif move_type == 1:
                # 2-for-1 swap: remove 2, add 1
                if len(code_set) < 3:
                    continue
                rem_list = list(code_set)
                c1, c2 = random.sample(rem_list, 2)
                c_add = random.randint(0, total - 1)

                if c_add in code_set:
                    continue

                b1 = balls[c1]
                b2 = balls[c2]
                b_add = balls[c_add]
                
                # Create masks
                in_add = np.zeros(total, dtype=np.uint8)
                in_add[b_add] = 1
                
                in_b1 = np.zeros(total, dtype=np.uint8)
                in_b1[b1] = 1
                
                in_b2 = np.zeros(total, dtype=np.uint8)
                in_b2[b2] = 1
                
                # For x in b1 U b2: need (cnt[x] - in_b1[x] - in_b2[x]) >= 1 + in_add[x]
                # i.e., cnt[x] >= 1 + in_add[x] + in_b1[x] + in_b2[x]
                
                # Check for x in b1
                vals1 = cnt[b1] - in_b1[b1] - in_b2[b1]
                need1 = 1 + in_add[b1]
                if np.any(vals1 < need1):
                    valid_move = False
                else:
                    # Check for x in b2 (only those not in b1 to avoid double counting, but we can just check all in b2)
                    vals2 = cnt[b2] - in_b1[b2] - in_b2[b2]
                    need2 = 1 + in_add[b2]
                    if np.any(vals2 < need2):
                        valid_move = False
                    else:
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(c_add)
                        cnt[b1] -= 1
                        cnt[b2] -= 1
                        cnt[b_add] += 1
                        valid_move = True

            else:
                # 1-for-2 swap: remove 1, add 2
                if len(code_set) < 1:
                    continue
                c_remove = random.choice(list(code_set))
                c_add1 = random.randint(0, total - 1)
                c_add2 = random.randint(0, total - 1)
                
                # Ensure distinct and not in code_set
                if c_add1 == c_add2 or c_add1 in code_set or c_add2 in code_set or c_add1 == c_remove or c_add2 == c_remove:
                    continue

                b_rem = balls[c_remove]
                b_add1 = balls[c_add1]
                b_add2 = balls[c_add2]
                
                # Create mask for b_add1 U b_add2
                in_add = np.zeros(total, dtype=np.uint8)
                in_add[b_add1] = 1
                in_add[b_add2] = 1
                
                # For x in b_rem: need (cnt[x] - 1) >= 1 + in_add[x]
                # i.e., cnt[x] >= 2 + in_add[x]
                vals = cnt[b_rem] - in_add[b_rem]
                if np.min(vals) >= 2:
                    code_set.remove(c_remove)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    cnt[b_rem] -= 1
                    cnt[b_add1] += 1
                    cnt[b_add2] += 1
                    valid_move = True

            if valid_move:
                no_success = 0
                try_delete()
            else:
                no_success += 1
                if no_success > max_streak:
                    # Reset streak, keep going
                    no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=12.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart
    seed = 0
    while time.time() < deadline - 8.0 and best_size_overall > 158:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        tb = min(10.0, max(1.5, time_left / 4.0))
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
