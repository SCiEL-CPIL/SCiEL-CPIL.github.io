import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin under 150s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Precompute base for index <-> word
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: list of lists of indices
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

    # Greedy construction with smarter candidate selection
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            
            # Only consider candidates that have at least one uncovered point in their ball
            # This is a heuristic to speed up convergence
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices[c]
                # Quick check: if all points in ball are covered, skip
                # Use numpy for fast counting
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

    # Local Search with focused moves
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
                # Process deletions in a stable order to avoid missing cascading deletions
                for c in sorted(code_set):
                    # Check if c is redundant: all points in its ball are covered by >= 2 codewords
                    if np.min(cnt[ball_indices[c]]) >= 2:
                        code_set.remove(c)
                        cnt[ball_indices[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        # Initial deletion pass
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

            elif move_type < 0.75:
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
                
                union_rem = set(ball_rem1).union(ball_rem2)
                add1_set = set(ball_add1)
                add2_set = set(ball_add2)
                
                valid = True
                for x in union_rem:
                    rem_cnt = (1 if x in ball_rem1 else 0) + (1 if x in ball_rem2 else 0)
                    add_cnt = (1 if x in add1_set else 0) + (1 if x in add2_set else 0)
                    
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
                # 2-for-1 swap (reduces size by 1)
                if len(code_set) < 2:
                    continue
                c_rem1, c_rem2 = random.sample(list(code_set), 2)
                c_add1 = random.randint(0, total - 1)
                
                if c_add1 in code_set:
                    continue
                
                ball_rem1 = ball_indices[c_rem1]
                ball_rem2 = ball_indices[c_rem2]
                ball_add1 = ball_indices[c_add1]
                
                union_rem = set(ball_rem1).union(ball_rem2)
                add_set = set(ball_add1)
                
                valid = True
                for x in union_rem:
                    rem_cnt = (1 if x in ball_rem1 else 0) + (1 if x in ball_rem2 else 0)
                    add_cnt = 1 if x in add_set else 0
                    
                    if cnt[x] - rem_cnt + add_cnt < 1:
                        valid = False
                        break
                
                if valid:
                    code_set.remove(c_rem1)
                    code_set.remove(c_rem2)
                    code_set.add(c_add1)
                    cnt[ball_rem1] -= 1
                    cnt[ball_rem2] -= 1
                    cnt[ball_add1] += 1
                    no_success_streak = 0
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    try_delete()
                else:
                    no_success_streak += 1

            if no_success_streak > max_streak:
                no_success_streak = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=10.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart with adaptive time allocation
    seed = 0
    while time.time() < deadline - 5.0 and best_size_overall > 160:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 5.0 - time.time()
        tb = min(6.0, max(0.5, time_left / 8.0))
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
