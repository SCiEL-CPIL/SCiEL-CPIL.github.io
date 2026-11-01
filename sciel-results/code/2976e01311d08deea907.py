import itertools
import numpy as np
import time

def construct(q, n, radius):
    # This task is specifically for q=3, n=7, radius=1
    # But we should handle the general interface if possible, 
    # though the prompt implies we can specialize if it helps, 
    # as long as we return valid codewords for the given params.
    # Given the 120s limit and the specific known instance, 
    # we will focus on q=3, n=7.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction for other cases
        # Direct sum of trivial codes
        # For radius 1, a trivial code is all words (q^n)
        # Or a product code.
        # Let's just return all words to be safe and valid, though score will be bad.
        # Or use the seed logic if it applies.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    total_words = 3**7
    # Precompute balls of radius 1 for all words
    # Word is an integer 0..2186, base 3 representation
    # Ball(w) = {w} U {w with 1 digit changed}
    
    # Convert integer to base 3 digits
    def int_to_base3(num, length=7):
        return [0] * length
    
    # We need balls for fast indexing.
    # balls[i] is an array of 15 integers.
    balls = np.zeros((total_words, 15), dtype=np.int32)
    
    # Precompute base-3 digits for all words
    digits = np.zeros((total_words, 7), dtype=np.int8)
    for i in range(total_words):
        num = i
        for j in range(6, -1, -1):
            digits[i, j] = num % 3
            num //= 3
    
    # Precompute balls
    for i in range(total_words):
        balls[i, 0] = i
        idx = 1
        for pos in range(7):
            orig_val = digits[i, pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                # Calculate new index
                # i + (new_val - orig_val) * (3**pos)
                # Note: digits[i, 0] is least significant (3^0), digits[i, 6] is most significant (3^6)
                # Wait, in the loop above:
                # j=6 -> num%3 -> digit 6. j=0 -> digit 0.
                # So digits[i, j] is the coefficient of 3^j.
                # So new_index = i + (new_val - orig_val) * (3**pos)
                balls[i, idx] = i + (new_val - orig_val) * (3**pos)
                idx += 1
        # idx should be 15
    
    # 1. Greedy Set Cover
    def greedy_cover():
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        # Precompute for each word, the list of words it covers (its ball)
        # We already have `balls`.
        
        while np.any(uncovered):
            best_c = -1
            best_count = 0
            # Iterate through all possible codewords
            # Optimization: We can check all 2187.
            for c in range(total_words):
                # Count how many uncovered words are in balls[c]
                # balls[c] is an array of indices
                count = np.sum(uncovered[balls[c]])
                if count > best_count:
                    best_count = count
                    best_c = c
                    if best_count == 15: # Max possible
                        break
            
            if best_c == -1:
                break # Should not happen if uncovered exists
            
            code.append(best_c)
            uncovered[balls[best_c]] = False
            
            if time.time() > deadline - 20:
                break
                
        return code

    # 2. Local Search
    def local_search(initial_code):
        code = list(initial_code)
        if not code:
            return code
            
        # Coverage counts
        cov = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cov[balls[c]] += 1
            
        best_code = code[:]
        best_size = len(code)
        
        # Helper to check if removing c is valid
        # We decrement cov[balls[c]], check min, then restore if needed
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Phase 1: Greedy Removal
            # Try to remove any redundant codeword
            # We iterate through the list. If we remove one, we restart the scan? 
            # Or just continue? Restarting is safer for finding all removals.
            # But O(N^2) might be slow. N is ~200. N^2 = 40k. Fine.
            
            removal_found = True
            while removal_found and time.time() < deadline - 5:
                removal_found = False
                for i in range(len(code)):
                    c = code[i]
                    # Tentative removal
                    ball = balls[c]
                    old_min = np.min(cov[ball])
                    cov[ball] -= 1
                    new_min = np.min(cov[ball])
                    
                    if new_min > 0:
                        # Safe to remove
                        code.pop(i)
                        improved = True
                        removal_found = True
                        break # Restart loop
                    else:
                        # Restore
                        cov[ball] += 1
                        
            if time.time() > deadline - 5:
                break
                
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                
            # Phase 2: Random Swaps to escape local minima
            # If no removal was found, try swaps
            # Pick a random codeword to remove
            # If invalid, add the best replacement
            # Repeat a few times
            
            swap_attempts = 50
            for _ in range(swap_attempts):
                if time.time() > deadline - 5:
                    break
                    
                if len(code) == 0:
                    break
                    
                idx_out = np.random.randint(len(code))
                c_out = code[idx_out]
                
                # Tentatively remove
                ball_out = balls[c_out]
                cov[ball_out] -= 1
                
                # Find uncovered words
                uncovered_mask = cov == 0
                if not np.any(uncovered_mask):
                    # Valid removal!
                    code.pop(idx_out)
                    improved = True
                    break # Go back to Phase 1
                else:
                    # Need to add a codeword to cover the uncovered words
                    # Find c_in that covers the most uncovered words
                    best_c_in = -1
                    best_cover = 0
                    # We can sample a few candidates or check all?
                    # Checking all 2187 is 2187 * 15 ops = 32k ops.
                    # 50 attempts * 32k = 1.6M ops. Fine.
                    for c_in in range(total_words):
                        if c_in in code: # Optimization: skip if already in code?
                            # Checking 'in' is O(N). N=200. 2187*200 = 400k.
                            # Maybe use a set for membership?
                            pass
                        cover = np.sum(uncovered_mask[balls[c_in]])
                        if cover > best_cover:
                            best_cover = cover
                            best_c_in = c_in
                            if best_cover == np.sum(uncovered_mask):
                                break
                                
                    if best_c_in != -1 and best_cover > 0:
                        # Add best_c_in
                        code.append(best_c_in)
                        cov[balls[best_c_in]] += 1
                    else:
                        # Restore c_out
                        cov[ball_out] += 1
                        break # No improvement possible with this c_out?
                        
                    # Note: Size remains same. We didn't improve size, but we changed the set.
                    # We set improved = False here, but we want to keep searching for removals.
                    # So we don't break the main while loop, but we do break the swap loop 
                    # and go back to Phase 1 to see if the new configuration allows removals.
                    
        return best_code

    # Run multiple restarts
    best_solution = None
    best_size = float('inf')
    
    # Restart 1: Greedy
    code1 = greedy_cover()
    if code1:
        optimized1 = local_search(code1)
        if len(optimized1) < best_size:
            best_size = len(optimized1)
            best_solution = optimized1
            
    if time.time() < deadline - 30:
        # Restart 2: Greedy with random tie-breaking
        # Modify greedy to pick randomly among best candidates
        def greedy_cover_random():
            uncovered = np.ones(total_words, dtype=bool)
            code = []
            while np.any(uncovered):
                best_count = 0
                candidates = []
                for c in range(total_words):
                    count = np.sum(uncovered[balls[c]])
                    if count > best_count:
                        best_count = count
                        candidates = [c]
                    elif count == best_count and count > 0:
                        candidates.append(c)
                
                if not candidates:
                    break
                best_c = np.random.choice(candidates)
                code.append(best_c)
                uncovered[balls[best_c]] = False
                
                if time.time() > deadline - 10:
                    break
            return code
            
        code2 = greedy_cover_random()
        if code2:
            optimized2 = local_search(code2)
            if len(optimized2) < best_size:
                best_size = len(optimized2)
                best_solution = optimized2
                
    if time.time() < deadline - 10:
        # Restart 3: Start from the 243-word seed and reduce
        # This is often very effective
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_code_ints = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=3):
                # Convert h + tail to integer
                val = 0
                for i, d in enumerate(h + list(tail)):
                    val += d * (3**i)
                seed_code_ints.append(val)
                
        optimized3 = local_search(seed_code_ints)
        if len(optimized3) < best_size:
            best_size = len(optimized3)
            best_solution = optimized3
            
    if best_solution is None:
        # Fallback
        best_solution = list(range(total_words))
        
    # Convert indices back to list of lists
    result = []
    for idx in best_solution:
        word = [0] * 7
        num = idx
        for i in range(7):
            word[i] = num % 3
            num //= 3
        result.append(word)
        
    return result
