import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Greedy Set Cover with NumPy optimization
    2. Aggressive Local Search with Pruning, Swaps, and Add-Remove moves
    3. Final Polish with intensive swap attempts
    """
    start_time = time.time()
    timeout = 115  # Safety margin from 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** 7
    n_words = total_words
    
    # Precompute digits and balls
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls as lists of indices
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)
        
    # Precompute ball matrix for fast greedy
    # ball_matrix[i, j] = 1 if j is in balls[i]
    ball_matrix = np.zeros((n_words, n_words), dtype=np.int8)
    for i in range(n_words):
        for j in balls[i]:
            ball_matrix[i, j] = 1

    # Precompute ball sets for faster membership testing in local search
    ball_sets = [set(b) for b in balls]

    def greedy_cover(seed):
        """Run greedy set cover with NumPy optimization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_words, dtype=np.int8)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            
            # Calculate coverage for all candidates
            # covs[i] = number of uncovered words covered by word i
            covs = ball_matrix.dot(uncovered)
            
            # Find max coverage
            max_cov = np.max(covs)
            if max_cov == 0:
                break
                
            # Find all candidates with max coverage
            best_indices = np.where(covs == max_cov)[0]
            best_idx = rng.choice(best_indices)
            
            code.append(best_idx)
            # Update uncovered
            uncovered[balls[best_idx]] = 0
                
        return code

    def local_search(code, seed, max_swap_attempts=200):
        """
        Apply local search with:
        1. Pruning
        2. Swaps
        3. Add-Remove (Grow then Prune)
        """
        if time.time() - start_time > timeout:
            return code
            
        count = np.zeros(n_words, dtype=np.int8)
        for c in code:
            count[balls[c]] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        # Helper to prune
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                # Check if any word in ball[c] has count < 2
                # If all have count >= 2, c is redundant
                min_count_in_ball = np.min(count[balls[c]])
                if min_count_in_ball >= 2:
                    count[balls[c]] -= 1
                    code_set.discard(c)
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed

        prune()
        
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            
            # 1. Prune
            if prune():
                improved = True
                continue
            
            # 2. Try Swaps - more aggressive
            swap_done = False
            for _ in range(max_swap_attempts):
                if time.time() - start_time > timeout:
                    return code
                if len(code) == 0:
                    break
                
                c_idx = rng.randint(0, len(code) - 1)
                c = code[c_idx]
                w = rng.randint(0, n_words - 1)
                
                if w == c or w in code_set:
                    continue
                    
                # Check validity: B(c) \ B(w) must be covered by others
                # Equivalent to: for all j in B(c), if j not in B(w), then count[j] >= 2
                # Or: min count in B(c) \ B(w) >= 2
                # Note: B(c) and B(w) are sets of indices.
                # We can use the count array.
                
                # Optimization: 
                # If B(c) subset B(w), then swap is always valid (count decreases in B(c), increases in B(w)).
                # But we need to check if removing c breaks coverage.
                # Words in B(c) that are NOT in B(w) must have count >= 2.
                
                # Using sets for intersection/difference is slow.
                # Using boolean masks is faster.
                # mask_c = np.zeros(n_words, dtype=bool); mask_c[balls[c]] = True
                # mask_w = np.zeros(n_words, dtype=bool); mask_w[balls[w]] = True
                # diff = mask_c & ~mask_w
                # if np.min(count[diff]) >= 2: valid
                
                # Precompute masks? 2187 * 2187 booleans = 4.7MB.
                # We have ball_matrix.
                # mask_c is row c of ball_matrix.
                # mask_w is row w of ball_matrix.
                
                row_c = ball_matrix[c]
                row_w = ball_matrix[w]
                
                # Find indices where row_c is 1 and row_w is 0
                # diff_indices = np.where(row_c & ~row_w)[0]
                
                # Instead of creating diff_indices, we can check:
                # min count in positions where row_c=1 and row_w=0
                # We can do:
                # temp_count = count.copy()
                # temp_count[row_w] = 2 # set covered by w to safe
                # if np.min(temp_count[row_c]) >= 2: valid
                # But copying count is slow.
                
                # Better:
                # Create a mask of "uncovered by w"
                # not_in_w = ~row_w
                # relevant = row_c & not_in_w
                # if np.any(relevant):
                #     if np.min(count[relevant]) < 2: invalid
                # else: valid (B(c) subset B(w))
                
                not_in_w = ~row_w
                relevant = row_c & not_in_w
                if np.any(relevant):
                    if np.min(count[relevant]) < 2:
                        continue
                # If no relevant, or min count >= 2, it's valid
                
                # Perform swap
                count[balls[c]] -= 1
                count[balls[w]] += 1
                code[c_idx] = w
                code_set.discard(c)
                code_set.add(w)
                improved = True
                swap_done = True
                break # Restart loop to prune
            
            if swap_done:
                continue
            
            # 3. Try Add-Remove
            if len(code) < n_words:
                w = rng.randint(0, n_words - 1)
                if w not in code_set:
                    # Save state to revert if needed
                    old_code = list(code)
                    old_count = count.copy()
                    old_set = set(code_set)
                    
                    # Add w
                    code.append(w)
                    code_set.add(w)
                    count[balls[w]] += 1
                    
                    # Prune
                    prune()
                    
                    # Check if size improved
                    if len(code) < len(old_code):
                        improved = True
                        # Keep the new state
                    else:
                        # Revert
                        code = old_code
                        count = old_count
                        code_set = old_set
                        
        return code

    def polish(code, seed, time_budget):
        """Final polish phase with many swap attempts."""
        if time.time() - start_time > timeout:
            return code
        
        count = np.zeros(n_words, dtype=np.int8)
        for c in code:
            count[balls[c]] += 1
            
        code_set = set(code)
        rng = random.Random(seed)
        
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                min_count_in_ball = np.min(count[balls[c]])
                if min_count_in_ball >= 2:
                    count[balls[c]] -= 1
                    code_set.discard(c)
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
            return changed
        
        prune()
        
        # Do many swap attempts
        attempts = 0
        while time.time() - start_time < time_budget:
            if len(code) == 0:
                break
            
            c_idx = rng.randint(0, len(code) - 1)
            c = code[c_idx]
            w = rng.randint(0, n_words - 1)
            
            if w == c or w in code_set:
                continue
            
            row_c = ball_matrix[c]
            row_w = ball_matrix[w]
            not_in_w = ~row_w
            relevant = row_c & not_in_w
            if np.any(relevant):
                if np.min(count[relevant]) < 2:
                    continue
            
            # Valid swap
            count[balls[c]] -= 1
            count[balls[w]] += 1
            code[c_idx] = w
            code_set.discard(c)
            code_set.add(w)
            prune()
            
            attempts += 1
            if attempts % 1000 == 0 and time.time() - start_time > timeout - 5:
                break
        
        return code

    best_code = None
    best_size = float('inf')
    
    num_trials = 10
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        seed = 100 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        code = local_search(code, seed, max_swap_attempts=150)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        # If we have time, run additional local search on the current best
        remaining = timeout - (time.time() - start_time)
        if best_size <= 210 and remaining > 15:
            extra_time = timeout - 10
            code = local_search(list(best_code), seed + 1000, max_swap_attempts=100)
            if len(code) < best_size:
                best_code = list(code)
                best_size = len(code)

    # Final polish on the best code with remaining time
    remaining = timeout - (time.time() - start_time)
    if best_code is not None and remaining > 5:
        polish_time = min(remaining - 3, 30)
        best_code = polish(best_code, 99999, start_time + polish_time + 3)
        best_size = len(best_code)
    
    if best_code is None:
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result
