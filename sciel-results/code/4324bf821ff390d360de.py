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
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            # Randomize start to break symmetry
            start = random.randint(0, total - 1)
            # We check all candidates. 
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices[c]
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
                    if np.min(cnt[ball_indices[c]]) >= 2:
                        code_set.remove(c)
                        cnt[ball_indices[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        no_success_streak = 0
        max_streak = 2000
        
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # 1-for-1 swap
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)
                
                if c_add in code_set or c_add == c_remove:
                    continue
                    
                ball_rem = ball_indices[c_remove]
                ball_add = ball_indices[c_add]
                
                valid = True
                add_set = set(ball_add)
                
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

            elif move_type < 0.8:
                # 2-for-2 swap
                if len(code_set) < 2:
                    continue
                c_rem1, c_rem2 = random.sample(list(code_set), 2)
                c_add1, c_add2 = random.sample(range(total), 2)
                
                if c_add1 in code_set or c_add2 in code_set:
                    continue
                if c_add1 == c_add2:
                    continue
                
                ball_rem1 = ball_indices[c_rem1]
                ball_rem2 = ball_indices[c_rem2]
                ball_add1 = ball_indices[c_add1]
                ball_add2 = ball_indices[c_add2]
                
                # Union of removed balls
                # We need to check if any point in (B(r1) U B(r2)) \ (B(a1) U B(a2)) has cnt < 2?
                # No.
                # For x in B(r1) U B(r2):
                # removed_count = (x in B(r1)) + (x in B(r2))
                # added_count = (x in B(a1)) + (x in B(a2))
                # new_cnt = cnt[x] - removed_count + added_count
                # We need new_cnt >= 1.
                
                valid = True
                # To speed up, we can just check the union of removed balls.
                # Union size <= 30.
                union_rem = set(ball_rem1).union(ball_rem2)
                
                for x in union_rem:
                    rem_cnt = 0
                    if x in ball_rem1: rem_cnt += 1
                    if x in ball_rem2: rem_cnt += 1
                    
                    add_cnt = 0
                    if x in ball_add1: add_cnt += 1
                    if x in ball_add2: add_cnt += 1
                    
                    if cnt[x] - rem_cnt + add_cnt < 1:
                        valid = False
                        break
                
                if valid:
                    code_set.remove(c_rem1)
                    code_set.remove(c_rem2)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    cnt[ball_rem1] -= 1
                    cnt[ball_rem2] -= 1
                    cnt[ball_add1] += 1
                    cnt[ball_add2] += 1
                    no_success_streak = 0
                    try_delete()
                else:
                    no_success_streak += 1

            else:
                # 3-for-2 swap (reduces size by 1)
                if len(code_set) < 3:
                    continue
                c_rem1, c_rem2, c_rem3 = random.sample(list(code_set), 3)
                c_add1, c_add2 = random.sample(range(total), 2)
                
                if c_add1 in code_set or c_add2 in code_set:
                    continue
                if c_add1 == c_add2:
                    continue
                
                ball_rem1 = ball_indices[c_rem1]
                ball_rem2 = ball_indices[c_rem2]
                ball_rem3 = ball_indices[c_rem3]
                ball_add1 = ball_indices[c_add1]
                ball_add2 = ball_indices[c_add2]
                
                valid = True
                union_rem = set(ball_rem1).union(ball_rem2).union(ball_rem3)
                
                for x in union_rem:
                    rem_cnt = 0
                    if x in ball_rem1: rem_cnt += 1
                    if x in ball_rem2: rem_cnt += 1
                    if x in ball_rem3: rem_cnt += 1
                    
                    add_cnt = 0
                    if x in ball_add1: add_cnt += 1
                    if x in ball_add2: add_cnt += 1
                    
                    if cnt[x] - rem_cnt + add_cnt < 1:
                        valid = False
                        break
                
                if valid:
                    code_set.remove(c_rem1)
                    code_set.remove(c_rem2)
                    code_set.remove(c_rem3)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    cnt[ball_rem1] -= 1
                    cnt[ball_rem2] -= 1
                    cnt[ball_rem3] -= 1
                    cnt[ball_add1] += 1
                    cnt[ball_add2] += 1
                    no_success_streak = 0
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    try_delete()
                else:
                    no_success_streak += 1

            if no_success_streak > max_streak:
                # Restart local search with current set? Or just continue.
                # Let's just reset streak to avoid infinite loop in bad regions
                no_success_streak = 0

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
    while time.time() < deadline - 5.0 and best_size_overall > 160:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 5.0 - time.time()
        tb = min(8.0, max(1.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
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
