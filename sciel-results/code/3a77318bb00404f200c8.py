import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # 1. Precompute Index <-> Word mapping
    # We need this to output the result.
    # For internal logic, we work with indices.
    
    # 2. Precompute the Incidence Matrix M
    # M[i, j] = 1 if index j is in the ball of index i (distance <= radius)
    # For radius 1, this is i itself and all single-digit changes.
    # This matrix is sparse, but for N=2187, a dense matrix (2187x2187 ~ 4.7M floats) 
    # is small enough to fit in cache and allows fast BLAS operations.
    
    # Base for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Create a dense matrix M
    # Using uint8 to save memory, but bool/int might be faster for dot products?
    # Let's use uint8 for storage, and cast to float32/int32 when needed for matrix mult if necessary.
    # Actually, for boolean logic, we can just use boolean arrays.
    
    # Efficient way to build M:
    # For each i, find j in B(i).
    # j = i + (v - orig) * base[coord]
    
    M = np.zeros((total, total), dtype=np.uint8)
    
    # Digits for all indices
    # digits[i] is a tuple/array of n digits
    # We can compute the ball for each i on the fly or precompute a list of balls.
    # Precomputing a list of balls (indices) is faster for construction.
    
    balls = []
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
        balls.append(ball)
        
    # Fill M
    for i, ball in enumerate(balls):
        M[i, ball] = 1
        
    # For fast "new coverage" calculation in greedy:
    # If `covered` is a boolean array (True if covered), 
    # `uncovered` is ~covered.
    # Number of new points covered by candidate c is sum(uncovered[j] for j in B(c)).
    # This is equivalent to (uncovered @ M.T)[c] ? 
    # Let's check: (A @ B)[i] = sum_k A[i,k] B[k,j].
    # We want S[c] = sum_{j in B(c)} uncovered[j].
    # Note that M[c, j] = 1 if j in B(c).
    # So S[c] = sum_j M[c, j] * uncovered[j].
    # This is the row c of M dot uncovered.
    # So `new_cov = M @ uncovered` (where uncovered is a column vector).
    
    # For Local Search:
    # cnt[x] = number of codewords covering x.
    # cnt = M.T @ code_indicator ? 
    # cnt[x] = sum_{c in Code} M[c, x].
    # Yes, cnt = M.T @ indicator(Code).
    
    # Optimization: Use float32 for matrix multiplications to be faster than int?
    # Or just stick with boolean/uint8 and use np.dot which handles it.
    
    # Let's convert M to float32 for faster dot products if needed, 
    # but for boolean masking, uint8 is fine.
    # Actually, np.dot on uint8 might be slower than float32.
    # Let's keep M as uint8 for storage, but for the "new_cov" calculation,
    # we can do: new_cov = np.dot(M, uncovered.astype(np.uint8))
    
    # Greedy Construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
            
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered_count = total
        uncovered = np.ones(total, dtype=np.uint8)
        
        while uncovered_count > 0:
            # Vectorized Greedy Step
            # new_cov[c] = number of uncovered points in B(c)
            # new_cov = M @ uncovered
            new_cov = np.dot(M, uncovered)
            
            # Find max
            max_new = np.max(new_cov)
            
            # If max_new == 0, we are stuck (should not happen if logic is right)
            if max_new == 0:
                break
                
            # Find all candidates with max_new
            candidates = np.where(new_cov == max_new)[0]
            
            # Random tie-breaking
            best_c = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_c)
            
            # Update covered and uncovered
            # covered[balls[best_c]] = 1
            # This is O(ball_size) which is 15. Very fast.
            b = balls[best_c]
            covered[b] = 1
            # Update uncovered_count efficiently?
            # We can just recalculate if needed, or track it.
            # For the next step, we need `uncovered` to be accurate.
            # uncovered = 1 - covered
            # Doing `uncovered = 1 - covered` is O(N) = 2187 ops.
            # Doing it 200 times is ~400k ops. Negligible.
            uncovered = 1 - covered
            uncovered_count = np.sum(uncovered)
            
        return code

    # Local Search
    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        # cnt[x] = coverage count of x
        # Initialize cnt
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
                # Check all codewords for redundancy
                # A codeword c is redundant if min(cnt[balls[c]]) >= 2
                # We can check this for all codewords in code_set.
                # To speed up, we can iterate and check.
                # Since k is small (~200), and ball size 15, this is 3000 ops per pass.
                for c in list(code_set):
                    # min of cnt in ball
                    if np.min(cnt[balls[c]]) >= 2:
                        code_set.remove(c)
                        cnt[balls[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                            
        def try_double_delete():
            nonlocal best_size, best_code
            if len(code_set) < 2:
                return False
            code_list = list(code_set)
            random.shuffle(code_list)
            # Limit checks
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
                    b1 = balls[c1]
                    b2 = balls[c2]
                    # Check validity
                    # For all x in B(c1) U B(c2), cnt[x] - [x in B(c1)] - [x in B(c2)] >= 1
                    # i.e., cnt[x] >= 1 + [x in B(c1)] + [x in B(c2)]
                    
                    valid = True
                    # Union of balls
                    # We can just check the union
                    # To avoid set union overhead, we can check b1 and b2 separately
                    # but handle overlap.
                    
                    # Let's just iterate over b1
                    for x in b1:
                        # x is in b1. Is it in b2?
                        # Checking `x in b2` is O(15).
                        # We can precompute a set for b2? Or just use the fact that balls are small.
                        in_b2 = 1 if x in b2 else 0
                        if cnt[x] < 1 + 1 + in_b2: # 1 (for c1) + 1 (for c2 if in b2) + 1 (need remaining)
                             # Wait.
                             # cnt[x] is total coverage.
                             # Removing c1 and c2 reduces coverage by (1 + in_b2).
                             # Remaining coverage = cnt[x] - (1 + in_b2).
                             # We need Remaining >= 1.
                             # So cnt[x] >= 2 + in_b2.
                             # If in_b2=0, cnt[x] >= 2.
                             # If in_b2=1, cnt[x] >= 3.
                             pass
                        if cnt[x] < 2 + in_b2:
                            valid = False
                            break
                    if not valid:
                        continue
                    for x in b2:
                        if x not in b1:
                            # in_b1 = 0
                            # Need cnt[x] >= 2 + 0 = 2
                            if cnt[x] < 2:
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
                        try_delete()
                        return True
            return False

        # Initial passes
        try_delete()
        try_double_delete()

        no_success_streak = 0
        max_streak = 2000
        
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break
            
            op_type = random.random()
            
            if op_type < 0.6:
                # Single swap
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)
                
                if c_add in code_set or c_add == c_remove:
                    no_success_streak += 1
                    continue
                    
                b_rem = balls[c_remove]
                b_add = balls[c_add]
                
                # Check validity:
                # For all x in B(rem), if x not in B(add), cnt[x] >= 2
                # (Because c_remove covers it, and we remove c_remove, so need at least 1 other)
                # If x in B(add), cnt[x] >= 1 (c_remove covers it, c_add covers it, so after swap, c_add covers it.
                # Wait.
                # Before: cnt[x] includes c_remove.
                # After: cnt[x] - 1 (remove c_rem) + 1 (add c_add) = cnt[x].
                # So coverage count doesn't change for x in B(rem) ∩ B(add)?
                # No.
                # Let's trace:
                # x in B(rem) \ B(add):
                #   Before: covered by c_rem.
                #   After: c_rem gone. c_add does NOT cover.
                #   Need: cnt[x] - 1 >= 1 => cnt[x] >= 2.
                # x in B(rem) ∩ B(add):
                #   Before: covered by c_rem.
                #   After: c_rem gone, c_add present.
                #   Net change: 0.
                #   Need: cnt[x] >= 1 (always true if covered).
                # x in B(add) \ B(rem):
                #   Before: not covered by c_rem.
                #   After: c_add present.
                #   Net change: +1.
                #   Always valid.
                
                # So we only need to check x in B(rem) \ B(add) for cnt[x] >= 2.
                
                valid = True
                # Optimization: Check intersection?
                # b_add_set = set(b_add)
                # for x in b_rem:
                #     if x not in b_add_set:
                #         if cnt[x] < 2: valid = False
                
                # Using numpy might be overkill for 15 elements.
                b_add_set = set(b_add)
                for x in b_rem:
                    if x not in b_add_set:
                        if cnt[x] < 2:
                            valid = False
                            break
                            
                if valid:
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[b_rem] -= 1
                    cnt[b_add] += 1
                    no_success_streak = 0
                    try_delete()
                else:
                    no_success_streak += 1
                    
            elif op_type < 0.85:
                # Double delete
                if len(code_set) >= 2:
                    success = try_double_delete()
                    if success:
                        no_success_streak = 0
                    else:
                        no_success_streak += 1
                else:
                    no_success_streak += 1
                    
            else:
                # Double swap
                if len(code_set) >= 2:
                    rem_list = list(code_set)
                    random.shuffle(rem_list)
                    c_rem1, c_rem2 = rem_list[0], rem_list[1]
                    
                    c_add1 = random.randint(0, total - 1)
                    c_add2 = random.randint(0, total - 1)
                    
                    if c_add1 in code_set or c_add2 in code_set or c_add1 == c_add2:
                        no_success_streak += 1
                        continue
                    
                    b_rem1 = balls[c_rem1]
                    b_rem2 = balls[c_rem2]
                    b_add1 = balls[c_add1]
                    b_add2 = balls[c_add2]
                    
                    # Validity:
                    # For all x in (B(rem1) U B(rem2)):
                    # New coverage = cnt[x] - [x in B(rem1)] - [x in B(rem2)] + [x in B(add1)] + [x in B(add2)]
                    # Need New coverage >= 1.
                    
                    add_set = set(b_add1) | set(b_add2)
                    valid = True
                    # Check union of b_rem1 and b_rem2
                    # To avoid double counting, we can iterate b_rem1, then b_rem2 \ b_rem1
                    b_rem1_set = set(b_rem1)
                    
                    for x in b_rem1:
                        in_rem1 = 1
                        in_rem2 = 1 if x in b_rem2 else 0
                        in_add = 1 if x in add_set else 0
                        # cnt[x] - in_rem1 - in_rem2 + in_add >= 1
                        if cnt[x] - in_rem1 - in_rem2 + in_add < 1:
                            valid = False
                            break
                    if not valid:
                        no_success_streak += 1
                        continue
                        
                    for x in b_rem2:
                        if x not in b_rem1_set:
                            in_rem1 = 0
                            in_rem2 = 1
                            in_add = 1 if x in add_set else 0
                            if cnt[x] - in_rem1 - in_rem2 + in_add < 1:
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

        # Final polish
        try_delete()
        try_double_delete()
        try_delete()
        
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
    while time.time() < deadline - 10.0 and best_size_overall > 160:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 10.0 - time.time()
        if time_left < 2.0:
            break
        tb = min(15.0, max(2.0, time_left / 3.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish
    if best_size_overall > 160 and time.time() < deadline - 5.0:
        polish_budget = min(10.0, deadline - 5.0 - time.time())
        if polish_budget > 1.0:
            best_code, best_size = local_search(best_code_overall, time_budget=polish_budget)
            if best_size < best_size_overall:
                best_size_overall = best_size
                best_code_overall = best_code

    # Convert to output format
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
