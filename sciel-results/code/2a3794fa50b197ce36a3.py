import time
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin under 120s

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base for index <-> word conversion later
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute ball_indices: list of lists of indices
    # Using pure Python lists for maximum speed in small-size operations
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
        
        covered = [0] * total
        code = []
        uncovered_count = total
        
        # List of all indices for shuffling
        all_indices = list(range(total))
        
        while uncovered_count > 0:
            best_c = -1
            max_new = -1
            
            # Randomize order to break symmetry
            random.shuffle(all_indices)
            
            for c in all_indices:
                ball = ball_indices[c]
                # Count uncovered points in this ball
                # Using a sum comprehension is fast for small lists
                new_cov = 0
                for x in ball:
                    if covered[x] == 0:
                        new_cov += 1
                
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            if best_c == -1:
                break # Should not happen if uncovered_count > 0

            code.append(best_c)
            ball = ball_indices[best_c]
            for x in ball:
                if covered[x] == 0:
                    covered[x] = 1
                    uncovered_count -= 1
            
        return code

    # Local Search
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        # cnt[x] is the number of codewords covering x
        cnt = [0] * total
        for c in code_set:
            for x in ball_indices[c]:
                cnt[x] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                # Iterate over a copy of the set to allow modification
                for c in list(code_set):
                    ball = ball_indices[c]
                    # Check if all points in this ball are covered by at least 2 codewords
                    # If so, this codeword is redundant
                    redundant = True
                    for x in ball:
                        if cnt[x] < 2:
                            redundant = False
                            break
                    
                    if redundant:
                        code_set.remove(c)
                        for x in ball:
                            cnt[x] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        # Initial delete pass
        try_delete()

        no_success_streak = 0
        max_streak = 1000
        
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break
            
            # 1-for-1 Swap
            # Pick a codeword to remove
            c_remove = random.choice(list(code_set))
            # Pick a candidate to add
            c_add = random.randint(0, total - 1)
            
            if c_add in code_set or c_add == c_remove:
                continue
                
            ball_rem = ball_indices[c_remove]
            ball_add = ball_indices[c_add]
            
            # Check validity
            # We need to ensure that any point covered ONLY by c_remove is also covered by c_add
            # i.e., if x in ball_rem and x not in ball_add, then cnt[x] must be >= 2
            # (cnt[x] includes c_remove, so if cnt[x] >= 2, there is at least one other codeword covering x)
            
            valid = True
            # Convert ball_add to a set for O(1) lookup
            add_set = set(ball_add)
            
            for x in ball_rem:
                if x not in add_set:
                    if cnt[x] < 2:
                        valid = False
                        break
            
            if valid:
                # Perform swap
                code_set.remove(c_remove)
                code_set.add(c_add)
                
                for x in ball_rem:
                    cnt[x] -= 1
                for x in ball_add:
                    cnt[x] += 1
                
                no_success_streak = 0
                
                # Try to delete redundant words after swap
                try_delete()
            else:
                no_success_streak += 1
                if no_success_streak > max_streak:
                    # Reset streak to keep trying
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
        # Allocate time for LS
        tb = min(8.0, max(1.0, time_left / 5.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final conversion to list of lists
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
