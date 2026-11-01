import time
import numpy as np
import random
import math

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin under 120s limit

    total = q ** n
    # ball_size = 1 + n * (q - 1)
    
    # Precompute base for index <-> word
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices and ball_sets
    ball_indices = [None] * total
    ball_sets = [None] * total
    
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
        ball_sets[i] = set(ball)

    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            # Randomize order to break symmetry
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices[c]
                new_cov = np.count_nonzero(covered[ball] == 0)
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == 1 + n * (q - 1): # Max possible gain
                        break
            
            code.append(best_c)
            covered[ball_indices[best_c]] = 1
            uncovered_count = np.count_nonzero(covered == 0)
            
        return code

    def simulated_annealing(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)
        
        # Initial temperature
        # Delta E is typically 0, 1, or -1
        # If we accept worse moves (Delta E = 1), we need T to be reasonable
        # Start T=1.0, cool to 0.1
        T = 1.0
        T_min = 0.1
        cooling_rate = 0.999 # Adjust based on iterations
        
        # Pre-compute list of code for random choice? 
        # We need to update it when code changes.
        
        # Helper to check if a swap is valid
        def is_valid_swap(rem, add):
            ball_rem = ball_indices[rem]
            ball_add_set = ball_sets[add]
            for x in ball_rem:
                if x not in ball_add_set:
                    if cnt[x] < 2:
                        return False
            return True

        def is_valid_double_swap(rem1, rem2, add1, add2):
            # Check if removing rem1, rem2 and adding add1, add2 keeps coverage >= 1
            # Points in (Ball(rem1) U Ball(rem2)) \ (Ball(add1) U Ball(add2)) must have cnt >= 2?
            # Actually, if a point is covered by rem1 and rem2, and neither add1 nor add2 covers it, 
            # its count drops by 2. So it must be >= 3?
            # No, cnt is total coverage.
            # If x in Ball(rem1) and x in Ball(rem2), cnt[x] includes 2 from these.
            # If x not in Ball(add1) and x not in Ball(add2), new cnt = cnt[x] - 2.
            # We need new cnt >= 1 => cnt[x] >= 3.
            
            # If x in Ball(rem1) only, and not in adds, new cnt = cnt[x] - 1 >= 1 => cnt[x] >= 2.
            
            # This check is more complex. Let's just check all points in Union(Rems) \ Union(Adds)
            # And calculate the drop.
            
            union_rems = ball_sets[rem1].union(ball_sets[rem2])
            union_adds = ball_sets[add1].union(ball_sets[add2])
            
            # We only care about points in union_rems that are NOT in union_adds
            # For each such point x, we need to know how many of rem1/rem2 cover it.
            # drop = (1 if x in Ball(rem1) else 0) + (1 if x in Ball(rem2) else 0)
            # We need cnt[x] - drop >= 1
            
            ball_rem1_set = ball_sets[rem1]
            ball_rem2_set = ball_sets[rem2]
            
            for x in union_rems:
                if x in union_adds:
                    continue
                drop = 0
                if x in ball_rem1_set: drop += 1
                if x in ball_rem2_set: drop += 1
                if cnt[x] - drop < 1:
                    return False
            return True

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

        # Main SA Loop
        while time.time() - t_start < time_budget:
            if T < T_min:
                break
                
            move_type = random.random()
            accepted = False
            delta_e = 0
            
            if move_type < 0.3:
                # Try Delete (always accepted if valid, delta_e = -1)
                # We do this in try_delete usually, but maybe we can try random deletion?
                # No, random deletion breaks coverage. Only redundant deletion is valid.
                # try_delete is called after successful moves.
                pass # Handled by try_delete below
            
            elif move_type < 0.7:
                # 1-for-1 Swap
                if len(code_set) == 0: continue
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)
                
                if c_add in code_set or c_add == c_remove:
                    continue
                
                if is_valid_swap(c_remove, c_add):
                    # Delta E = 0 (size unchanged)
                    # We accept this to move in the landscape
                    # Or we can treat it as neutral
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[ball_indices[c_remove]] -= 1
                    cnt[ball_indices[c_add]] += 1
                    accepted = True
                    delta_e = 0
                else:
                    # Invalid swap, reject
                    pass

            elif move_type < 0.9:
                # 2-for-1 Swap (Reduce size by 1)
                if len(code_set) < 2: continue
                c_rem1, c_rem2 = random.sample(list(code_set), 2)
                c_add1 = random.randint(0, total - 1)
                
                if c_add1 in code_set:
                    continue
                
                # Check validity
                # Points in (Ball(rem1) U Ball(rem2)) \ Ball(add1) must have cnt >= drop + 1
                union_rems = ball_sets[c_rem1].union(ball_sets[c_rem2])
                ball_add_set = ball_sets[c_add1]
                ball_rem1_set = ball_sets[c_rem1]
                ball_rem2_set = ball_sets[c_rem2]
                
                valid = True
                for x in union_rems:
                    if x in ball_add_set:
                        continue
                    drop = 0
                    if x in ball_rem1_set: drop += 1
                    if x in ball_rem2_set: drop += 1
                    if cnt[x] - drop < 1:
                        valid = False
                        break
                
                if valid:
                    code_set.remove(c_rem1)
                    code_set.remove(c_rem2)
                    code_set.add(c_add1)
                    cnt[ball_indices[c_rem1]] -= 1
                    cnt[ball_indices[c_rem2]] -= 1
                    cnt[ball_indices[c_add1]] += 1
                    accepted = True
                    delta_e = -1 # Improvement!
                else:
                    # Invalid, reject
                    pass

            else:
                # 1-for-2 Swap (Increase size by 1) - Needed to escape local minima
                if len(code_set) == 0: continue
                c_rem1 = random.choice(list(code_set))
                c_add1, c_add2 = random.sample(range(total), 2)
                
                if c_add1 in code_set or c_add2 in code_set:
                    continue
                if c_add1 == c_add2:
                    continue
                
                # Check validity
                # Points in Ball(rem1) \ (Ball(add1) U Ball(add2)) must have cnt >= 2
                ball_rem_set = ball_sets[c_rem1]
                union_adds = ball_sets[c_add1].union(ball_sets[c_add2])
                
                valid = True
                for x in ball_rem_set:
                    if x in union_adds:
                        continue
                    if cnt[x] < 2:
                        valid = False
                        break
                
                if valid:
                    code_set.remove(c_rem1)
                    code_set.add(c_add1)
                    code_set.add(c_add2)
                    cnt[ball_indices[c_rem1]] -= 1
                    cnt[ball_indices[c_add1]] += 1
                    cnt[ball_indices[c_add2]] += 1
                    accepted = True
                    delta_e = 1 # Worse, but might lead to better
                else:
                    pass

            if accepted:
                # If we improved, reset T? Or just keep cooling?
                # If delta_e == -1, we found a better solution.
                if delta_e < 0:
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    # Reheat slightly? Or just continue.
                    # try_delete might find more reductions
                    try_delete()
                
                # Cooling
                T *= cooling_rate
            
            # Optional: If stuck for a long time, reheat
            # For simplicity, we rely on multiple restarts

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # Run multiple restarts
    seed = 42
    while time.time() < deadline - 5.0:
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 5.0 - time.time()
        if time_left < 2.0:
            break
        tb = min(10.0, time_left / 2.0) # Spend up to 10s on SA
        best_code, best_size = simulated_annealing(greedy_code, time_budget=tb)
        
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code
        
        seed += 1

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
