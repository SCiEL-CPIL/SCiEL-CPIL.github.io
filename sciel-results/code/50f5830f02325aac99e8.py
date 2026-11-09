import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Greedy Set Cover with randomization
    2. Aggressive Local Search with Pruning, Swaps, and Add-Remove moves
    3. Final Polish with intensive swap attempts
    """
    start_time = time.time()
    timeout = 118  # Safety margin from 120s limit
    
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
    
    # Precompute ball sets for faster membership testing
    ball_sets = [set(b) for b in balls]
    
    def greedy_cover(seed):
        """Run greedy set cover with randomization and lazy evaluation."""
        rng = random.Random(seed)
        uncovered = np.ones(n_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            
            # Find best candidate by counting uncovered neighbors
            best_idx = -1
            max_cov = 0
            best_candidates = []
            
            for i in range(n_words):
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                        if cov > max_cov:
                            pass
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                    if max_cov == 15:
                        break
                elif cov == max_cov and max_cov > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            best_idx = rng.choice(best_candidates)
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
                
        return code
    
    def local_search(code, seed, max_swap_attempts=500):
        """
        Apply local search with:
        1. Pruning
        2. Swaps
        3. Add-Remove (Grow then Prune)
        """
        if time.time() - start_time > timeout:
            return code
            
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        # Helper to prune
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                redundant = True
                for j in balls[c]:
                    if count[j] < 2:
                        redundant = False
                        break
                if redundant:
                    for j in balls[c]:
                        count[j] -= 1
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
                ball_w = ball_sets[w]
                valid = True
                for j in balls[c]:
                    if j not in ball_w and count[j] < 2:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    for j in balls[c]:
                        count[j] -= 1
                    for j in balls[w]:
                        count[j] += 1
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
                    for j in balls[w]:
                        count[j] += 1
                    
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
        
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        def prune():
            nonlocal code, count, code_set
            new_code = []
            changed = False
            for c in code:
                redundant = True
                for j in balls[c]:
                    if count[j] < 2:
                        redundant = False
                        break
                if redundant:
                    for j in balls[c]:
                        count[j] -= 1
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
            
            ball_w = ball_sets[w]
            valid = True
            for j in balls[c]:
                if j not in ball_w and count[j] < 2:
                    valid = False
                    break
            
            if valid:
                for j in balls[c]:
                    count[j] -= 1
                for j in balls[w]:
                    count[j] += 1
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
    
    num_trials = 20
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        seed = 100 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        code = local_search(code, seed, max_swap_attempts=500)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        # If we have time, run additional local search on the current best
        remaining = timeout - (time.time() - start_time)
        if best_size <= 200 and remaining > 15:
            extra_time = timeout - 10
            code = local_search(list(best_code), seed + 1000, max_swap_attempts=300)
            if len(code) < best_size:
                best_code = list(code)
                best_size = len(code)
    
    # Final polish on the best code with remaining time
    remaining = timeout - (time.time() - start_time)
    if best_code is not None and remaining > 5:
        polish_time = min(remaining - 3, 40)
        best_code = polish(best_code, 99999, start_time + polish_time + 3)
        best_size = len(best_code)
    
    if best_code is None:
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result
