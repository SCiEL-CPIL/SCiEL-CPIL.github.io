import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by Local Search 
    with pruning and valid swaps.
    """
    # Deadline guard
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
    # digits[i] is a tuple of 7 ints (0,1,2) representing word i
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls: list of lists of indices
    # ball[i] contains all j such that dist(i,j) <= 1
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
        
    # Convert balls to numpy arrays for faster operations if needed, 
    # but lists are fine for small sizes (15 elements)
    
    # 1. Greedy Set Cover
    uncovered = np.ones(n_words, dtype=bool)
    code = []
    
    # Precompute ball masks as boolean arrays? 
    # Or just use the list of indices.
    # For greedy, we need to pick the center that covers most uncovered points.
    # We can iterate all centers, count how many uncovered points are in their ball.
    
    # Optimization: 
    # Instead of O(N^2) per step, we can just do it. 
    # N=2187. Steps ~ 200. 200 * 2187 * 15 ~ 6.5M ops. Very fast.
    
    while np.any(uncovered):
        if time.time() - start_time > timeout:
            break
        best_idx = -1
        max_cov = 0
        for i in range(n_words):
            # Count uncovered in ball[i]
            # We can sum uncovered[j] for j in balls[i]
            # Using numpy might be faster if we precomputed ball masks, 
            # but let's stick to simple loop or vectorized sum on small list
            cov = 0
            for j in balls[i]:
                if uncovered[j]:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_idx = i
                # Small optimization: if max_cov == 15 (full ball), we can't do better
                if max_cov == 15: 
                    break
        
        if best_idx == -1:
            break
            
        code.append(best_idx)
        # Update uncovered
        for j in balls[best_idx]:
            uncovered[j] = False
            
    # Initial code size is likely ~200-220
    
    # 2. Local Search
    # Maintain coverage counts
    count = np.zeros(n_words, dtype=int)
    for c in code:
        for j in balls[c]:
            count[j] += 1
            
    # Helper to check if a codeword c is redundant
    # c is redundant if for all j in balls[c], count[j] >= 2
    # (Because removing c reduces count[j] by 1, so it must remain >= 1)
    
    # Helper to check if swap c -> w is valid
    # Valid if for all j in balls[c] \ balls[w], count[j] >= 2
    # (Because removing c reduces count, adding w increases count for balls[w].
    #  Net change for j in balls[c] \ balls[w] is -1. Must stay >= 1 => old count >= 2)
    
    # Precompute set of balls for faster intersection/difference?
    # balls are small lists.
    
    rng = random.Random(42)
    
    # Local search loop
    while time.time() - start_time < timeout:
        
        # Step A: Prune
        pruned = False
        # Check each codeword
        # We can iterate through code and remove if redundant
        # To avoid index issues, we can build a new list
        new_code = []
        for c in code:
            # Check if c is redundant
            redundant = True
            for j in balls[c]:
                if count[j] < 2:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for j in balls[c]:
                    count[j] -= 1
                pruned = True
            else:
                new_code.append(c)
        
        code = new_code
        
        # Step B: Swap
        if not pruned:
            # If no pruning happened, try a swap
            # Pick a random codeword c
            if len(code) == 0:
                break
            c_idx = rng.randint(0, len(code) - 1)
            c = code[c_idx]
            
            # Pick a random word w from the whole space
            # To make it more effective, we could pick w from words that are "far" or random
            w = rng.randint(0, n_words - 1)
            
            if w == c:
                continue
                
            # Check if w is already in code
            # Using a set for code is faster for lookup, but we need to update it
            # Let's maintain a set of code words for O(1) lookup
            # Actually, checking if w in code (list) is O(N). N~200. 200 checks.
            # We do this many times. Let's use a set.
            
            # We need to update the set if we accept.
            # Let's maintain code_set alongside code list.
            
            # For now, just check list.
            if w in code:
                continue
                
            # Check validity of swap c -> w
            # Condition: for all j in balls[c] that are NOT in balls[w], count[j] >= 2
            # balls[w] is a list. Converting to set is O(15).
            ball_w_set = set(balls[w])
            
            valid = True
            for j in balls[c]:
                if j not in ball_w_set:
                    if count[j] < 2:
                        valid = False
                        break
                        
            if valid:
                # Accept swap
                # Update count
                # Remove c
                for j in balls[c]:
                    count[j] -= 1
                # Add w
                for j in balls[w]:
                    count[j] += 1
                # Update code
                code[c_idx] = w
                
    # Convert to output format
    result = [list(digits[c]) for c in code]
    return result
