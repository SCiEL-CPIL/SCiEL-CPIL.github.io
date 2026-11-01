import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin (150s kill, leave 40s buffer)

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Precompute base for index <-> word
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: list of arrays of indices
    ball_indices = np.empty((total, ball_size), dtype=np.int32)
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

    # Precompute ball sets for fast membership
    ball_sets = [set(ball_indices[i]) for i in range(total)]

    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        while uncovered_count > 0:
            # Vectorized: for each candidate c, count uncovered in its ball
            # covered[ball_indices] gives shape (total, ball_size)
            ball_cov = covered[ball_indices]  # (total, ball_size)
            new_cov = np.sum(ball_cov == 0, axis=1)  # (total,)
            
            best_c = int(np.argmax(new_cov))
            max_new = int(new_cov[best_c])
            
            if max_new == 0:
                break  # Should not happen if uncovered_count > 0
            
            code.append(best_c)
            covered[ball_indices[best_c]] = 1
            uncovered_count = int(np.sum(covered == 0))
            
            if time.time() > deadline - 10.0:
                break

        return code

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

        # Initial delete pass
        try_delete()

        no_success_streak = 0
        max_streak = 500
        
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break
            
            # Pick a codeword to remove
            c_remove = random.choice(list(code_set))
            # Pick a codeword to add
            c_add = random.randint(0, total - 1)
            
            if c_add in code_set or c_add == c_remove:
                continue
                
            ball_rem = ball_indices[c_remove]
            ball_add = ball_indices[c_add]
            
            # Check validity: any point in ball_rem not in ball_add must have cnt >= 2
            add_set = ball_sets[c_add]
            
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
                if no_success_streak > max_streak:
                    no_success_streak = 0
                    max_streak = int(max_streak * 1.5)

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=12.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart with vectorized greedy
    seed = 0
    while time.time() < deadline - 8.0 and best_size_overall > 162:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        tb = min(10.0, max(1.5, time_left / 4.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish: try more aggressive local search on best solution
    if time.time() < deadline - 3.0 and best_size_overall > 162:
        time_left = deadline - 3.0 - time.time()
        best_code, best_size = local_search(best_code_overall, time_budget=time_left)
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
