import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Specific parameters for the task
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius, deadline)
        
    total_words = 3 ** n # 2187
    
    # Precomputation
    # 1. idx to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
        
    # 2. word to idx
    word_to_idx = {}
    for i in range(total_words):
        word_to_idx[tuple(idx_to_word(i))] = i
        
    # 3. Balls as bitmasks and lists
    balls_mask = [0] * total_words
    balls_list = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        balls_list[i].append(i)
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                nidx = word_to_idx[tuple(w_list)]
                mask |= (1 << nidx)
                balls_list[i].append(nidx)
                w_list[pos] = orig_val
        balls_mask[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    best_code_size = total_words + 1
    best_code = None
    
    # Multi-start
    max_restarts = 50 # Adjust based on speed
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # --- Greedy Phase ---
        uncovered = full_mask
        code = []
        in_code = [False] * total_words
        
        while uncovered:
            best_cand = -1
            best_cov = -1
            # Random tie-breaking
            # Iterate all candidates
            for c in range(total_words):
                if in_code[c]:
                    continue
                cov = (balls_mask[c] & uncovered).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_cand = c
                elif cov == best_cov and cov > 0:
                    # Random tie break
                    if random.random() < 0.5:
                        best_cand = c
            
            if best_cand == -1:
                break # Should not happen
            
            code.append(best_cand)
            in_code[best_cand] = True
            uncovered &= ~balls_mask[best_cand]
            
        # --- Local Search Phase ---
        # Initialize cover_count
        cover_count = np.zeros(total_words, dtype=np.uint8)
        for c in code:
            for x in balls_list[c]:
                cover_count[x] += 1
        
        # Set of indices currently in code for O(1) lookup
        code_set = set(code)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to delete
            # Shuffle code to randomize order
            code_list = code[:]
            random.shuffle(code_list)
            
            for i, c in enumerate(code_list):
                if c not in code_set:
                    continue # Already removed
                
                # Check if c is redundant
                # Find words covered ONLY by c
                redundant = True
                for x in balls_list[c]:
                    if cover_count[x] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    code.remove(c)
                    code_set.remove(c)
                    for x in balls_list[c]:
                        cover_count[x] -= 1
                    improved = True
                    break # Restart deletion loop
            
            if not improved:
                # Try to swap
                # Pick a random codeword
                if code:
                    c_i = random.choice(code)
                    # Find words covered ONLY by c_i
                    only_c_i_mask = 0
                    for x in balls_list[c_i]:
                        if cover_count[x] == 1:
                            only_c_i_mask |= (1 << x)
                    
                    if only_c_i_mask != 0:
                        # Find a replacement c_new not in code_set
                        # that covers only_c_i_mask
                        found_swap = False
                        # To speed up, we can pre-filter candidates?
                        # For now, iterate all
                        for c_new in range(total_words):
                            if c_new in code_set:
                                continue
                            if (balls_mask[c_new] & only_c_i_mask) == only_c_i_mask:
                                # Valid swap
                                # Perform swap
                                code.remove(c_i)
                                code_set.remove(c_i)
                                for x in balls_list[c_i]:
                                    cover_count[x] -= 1
                                    
                                code.append(c_new)
                                code_set.add(c_new)
                                for x in balls_list[c_new]:
                                    cover_count[x] += 1
                                
                                improved = True
                                found_swap = True
                                break
                        # If no swap found, we are stuck in this local minimum
                        # We can break out of the swap attempt for this restart
                        # or try another c_i? 
                        # Trying another c_i is cheap if we just check.
                        # But if the whole code is "rigid", we need to restart.
                        # Let's break to restart greedy with different noise.
                        if not found_swap:
                            break
                    else:
                        # c_i is redundant? Should have been deleted.
                        # Remove it
                        code.remove(c_i)
                        code_set.remove(c_i)
                        for x in balls_list[c_i]:
                            cover_count[x] -= 1
                        improved = True

        if len(code) < best_code_size:
            best_code_size = len(code)
            best_code = code[:]
            
        # Optional: If we found a very good one, we can try to polish it more?
        # The loop above already polishes until stuck.
        
    # Convert best_code to list of lists
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result

def _generic_greedy(q, n, radius, deadline):
    total_words = q ** n
    if total_words > 100000:
         return [] # Fallback for huge spaces, though not expected
    
    # Simple greedy
    uncovered = set(range(total_words))
    selected = []
    
    # Precompute balls
    balls = []
    for i in range(total_words):
        w = [0]*n
        tmp = i
        for k in range(n):
            w[k] = tmp % q
            tmp //= q
        w = tuple(w)
        ball = set()
        ball.add(i)
        for pos in range(n):
            for val in range(q):
                if w[pos] == val: continue
                nw = list(w)
                nw[pos] = val
                nidx = 0
                for k in range(n):
                    nidx = nidx * q + nw[k]
                ball.add(nidx)
        balls.append(ball)
        
    while uncovered:
        best_c = -1
        best_cov = -1
        for c in range(total_words):
            cov = len(balls[c] & uncovered)
            if cov > best_cov:
                best_cov = cov
                best_c = c
        if best_c == -1: break
        selected.append(best_c)
        uncovered -= balls[best_c]
        
    result = []
    for idx in selected:
        w = [0]*n
        tmp = idx
        for k in range(n):
            w[k] = tmp % q
            tmp //= q
        result.append(w)
    return result
