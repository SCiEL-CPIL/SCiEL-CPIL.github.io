import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Precompute base for index <-> word
    # Not strictly needed if we just use indices, but needed for output
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: list of lists of indices
    # ball_indices[i] is a list of 15 indices
    ball_indices = [None] * total
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
        ball_indices[i] = ball

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        # Optimize greedy: 
        # Instead of scanning all 2187 every time (2187 * 15 ops * 200 iters ~ 6.5M ops),
        # we can just do it. It's fast enough.
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            # Randomize start to break symmetry
            start = random.randint(0, total - 1)
            # We check all candidates. 
            # To speed up, we can skip candidates that are already covered? 
            # No, a covered candidate might cover *other* uncovered points.
            
            # Optimization: Only check candidates that cover at least 1 uncovered point?
            # Actually, if a candidate's ball is fully covered, new_cov is 0.
            # We can precompute which candidates are "active".
            
            # Simple loop
            # Use a list of all indices and shuffle? 
            # Shuffling 2187 items is fast.
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices[c]
                # Count uncovered in ball
                # covered is a numpy array. 
                # new_cov = sum(covered[x] == 0 for x in ball)
                # Using numpy:
                new_cov = np.count_nonzero(covered[ball] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            code.append(best_c)
            covered[ball_indices[best_c]] = 1
            uncovered_count = np.count_nonzero(covered == 0)
            
        return code

    # Local Search
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    # Check if c is redundant
                    # min(cnt[ball_indices[c]]) >= 2
                    if np.min(cnt[ball_indices[c]]) >= 2:
                        code_set.remove(c)
                        cnt[ball_indices[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        def try_double_delete():
            """Try to remove two codewords at once if their combined coverage is redundant."""
            nonlocal best_size, best_code
            if len(code_set) < 2:
                return
            code_list = list(code_set)
            random.shuffle(code_list)
            # Limit checks to avoid O(k^2) blowup
            max_checks = min(50, len(code_list) * 2)
            checked = 0
            for i in range(min(len(code_list), 20)):
                for j in range(i + 1, min(len(code_list), 20)):
                    if checked >= max_checks:
                        break
                    checked += 1
                    c1, c2 = code_list[i], code_list[j]
                    if c1 not in code_set or c2 not in code_set:
                        continue
                    b1 = ball_indices[c1]
                    b2 = ball_indices[c2]
                    # Check: for all x in (B(c1) ∪ B(c2)), cnt[x] >= 3 (so removing both still leaves coverage)
                    # Actually we need: for all x in B(c1) ∪ B(c2), cnt[x] - [x in B(c1)] - [x in B(c2)] >= 1
                    valid = True
                    for x in b1:
                        if cnt[x] - 1 < 1:  # cnt[x] < 2, but x is in b1 so at least c1 covers it
                            valid = False
                            break
                        # Also need to check if x is also in b2
                        if x in b2 and cnt[x] - 2 < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    for x in b2:
                        if x not in b1:  # already checked if in b1
                            if cnt[x] - 1 < 1:
                                valid = False
                                break
                            if x in b1 and cnt[x] - 2 < 1:
                                valid = False
                                break
                    if valid:
                        code_set.remove(c1)
                        code_set.remove(c2)
                        cnt[b1] -= 1
                        cnt[b2] -= 1
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                        # Try single deletes after double delete
                        try_delete()
                        return True
            return False

        # Initial delete pass
        try_delete()
        try_double_delete()

        # Swap loop with single and double swaps
        no_success_streak = 0
        max_streak = 2000
        
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break
            
            op_type = random.random()
            
            if op_type < 0.6:
                # Single swap: remove one, add one
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)
                
                if c_add in code_set or c_add == c_remove:
                    no_success_streak += 1
                    continue
                    
                ball_rem = ball_indices[c_remove]
                ball_add = ball_indices[c_add]
                
                add_set = set(ball_add)
                valid = True
                for x in ball_rem:
                    if x not in add_set:
                        if cnt[x] < 2:
                            valid = False
                            break
                
                if valid:
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[ball_rem] -= 1
                    cnt[ball_add] += 1
                    no_success_streak = 0
                    try_delete()
                else:
                    no_success_streak += 1
                    
            elif op_type < 0.85:
                # Double delete: remove two codewords (no addition)
                if len(code_set) >= 2:
                    success = try_double_delete()
                    if success:
                        no_success_streak = 0
                    else:
                        no_success_streak += 1
                else:
                    no_success_streak += 1
                    
            else:
                # Double swap: remove two, add two
                if len(code_set) >= 2:
                    rem_list = list(code_set)
                    random.shuffle(rem_list)
                    c_rem1, c_rem2 = rem_list[0], rem_list[1]
                    
                    c_add1 = random.randint(0, total - 1)
                    c_add2 = random.randint(0, total - 1)
                    
                    if c_add1 in code_set or c_add2 in code_set or c_add1 == c_add2:
                        no_success_streak += 1
                        continue
                    
                    b_rem1 = ball_indices[c_rem1]
                    b_rem2 = ball_indices[c_rem2]
                    b_add1 = ball_indices[c_add1]
                    b_add2 = ball_indices[c_add2]
                    
                    # Check validity: for all x in (B(rem1) ∪ B(rem2)), after removal, x must be covered by add1 or add2 or remaining codewords
                    add_set = set(b_add1) | set(b_add2)
                    valid = True
                    for x in b_rem1:
                        if x not in add_set:
                            # x must be covered by at least one other codeword (cnt[x] >= 2 before removal)
                            # But if x is also in b_rem2, cnt[x] >= 3 needed
                            if x in b_rem2:
                                if cnt[x] < 3:
                                    valid = False
                                    break
                            else:
                                if cnt[x] < 2:
                                    valid = False
                                    break
                    if valid:
                        for x in b_rem2:
                            if x not in b_rem1:  # already checked overlap
                                if x not in add_set:
                                    if cnt[x] < 2:
                                        valid = False
                                        break
                    if valid:
                        code_set.remove(c_rem1)
                        code_set.remove(c_rem2)
                        code_set.add(c_add1)
                        code_set.add(c_add2)
                        cnt[b_rem1] -= 1
                        cnt[b_rem2] -= 1
                        cnt[b_add1] += 1
                        cnt[b_add2] += 1
                        no_success_streak = 0
                        try_delete()
                        try_double_delete()
                    else:
                        no_success_streak += 1
                else:
                    no_success_streak += 1
            
            if no_success_streak > max_streak:
                no_success_streak = 0

        # Final polish: aggressive deletion
        try_delete()
        try_double_delete()
        try_delete()
        
        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=12.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart with adaptive time budget
    seed = 0
    while time.time() < deadline - 8.0 and best_size_overall > 160:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        if time_left < 2.0:
            break
        # Allocate time: more if we're close to good solutions
        tb = min(10.0, max(1.5, time_left / 4.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish on best solution found
    if best_size_overall > 160 and time.time() < deadline - 3.0:
        polish_budget = min(5.0, deadline - 3.0 - time.time())
        if polish_budget > 1.0:
            best_code, best_size = local_search(best_code_overall, time_budget=polish_budget)
            if best_size < best_size_overall:
                best_size_overall = best_size
                best_code_overall = best_code

    # Final conversion
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
