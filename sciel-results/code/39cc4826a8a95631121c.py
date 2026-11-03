import numpy as np
import time
import random

def construct(q, n, radius):
    if not (q == 3 and n == 6 and radius == 1):
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = 115.0
    
    N = q ** n
    # N = 729

    # 1. Precomputation
    # Generate all words as indices 0..728
    # idx = w0 * 3^5 + w1 * 3^4 + ... + w5
    
    # Precompute neighbors for each index
    # neighbor_matrix[i] is a list of indices within distance 1
    # We store this as a list of arrays for fast iteration? 
    # Or just a flat array?
    # Let's use a list of lists for clarity, but optimized.
    
    balls = [None] * N
    for i in range(N):
        w = np.array([0]*n)
        idx = i
        for j in range(n-1, -1, -1):
            w[j] = idx % q
            idx //= q
        
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    new_idx = 0
                    for val in new_w:
                        new_idx = new_idx * q + val
                    ball.append(new_idx)
        balls[i] = np.array(ball)

    # Precompute "coverers" is expensive (729 * 19 * 729 checks).
    # Instead, we will maintain coverage counts dynamically.
    
    # 2. Initial Solution: Greedy
    # Use NumPy to speed up "cover most uncovered"
    
    # uncovered: boolean array of size N
    uncovered = np.ones(N, dtype=bool)
    cover_counts = np.zeros(N, dtype=np.int32)
    code = []
    
    # Greedy Phase
    # To speed up, we can precompute the "value" of adding a word?
    # Value of word i = sum(uncovered[balls[i]])
    # We can update this incrementally?
    # Simpler: Just iterate. 729 * 729 is small.
    
    while np.any(uncovered):
        if time.time() > deadline - 10: break
        
        best_cw = -1
        best_gain = -1
        
        # Optimization: Only consider words that are "good"
        # We can sample or just iterate all. 729 is small.
        for cw in range(N):
            if cw in code: continue # Set lookup is slow, use boolean array
            # Actually, let's use a boolean array 'in_code'
            
        # Let's use a set for code membership for simplicity in greedy, 
        # but for local search we need speed.
        
        # Revised Greedy with NumPy
        in_code = np.zeros(N, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline - 20: break
            
            # Calculate gain for each candidate
            # gain[cw] = sum(uncovered[balls[cw]])
            # This is 729 * 19 ops = ~14k ops. Very fast.
            
            gains = np.zeros(N, dtype=np.int32)
            for cw in range(N):
                if in_code[cw]: continue
                # Sum uncovered in ball
                gains[cw] = np.sum(uncovered[balls[cw]])
                
            # Find max
            best_cw = np.argmax(gains)
            
            if gains[best_cw] == 0:
                break # No more coverage possible? Should not happen if valid space
                
            code.append(best_cw)
            in_code[best_cw] = True
            cover_counts[balls[best_cw]] += 1
            uncovered[balls[best_cw]] = False # Wait, uncovered should be cover_counts == 0
            
            # Update uncovered
            uncovered = (cover_counts == 0)
            
    # Initial code size is likely 74-80.
    
    # 3. Local Search
    # State: code (list), cover_counts (array), in_code (bool array)
    
    def try_remove_and_swap(code_list, cover_counts, in_code, balls, deadline):
        # Try to remove each word and re-optimize
        # We will do random swaps to escape local minima
        
        best_size = len(code_list)
        best_code = list(code_list)
        best_cover = cover_counts.copy()
        best_in = in_code.copy()
        
        stuck = 0
        
        while time.time() < deadline:
            # 1. Greedy Reduction
            # Try to remove any redundant word
            removed = True
            while removed:
                removed = False
                # Pick a random word to check
                # Or iterate?
                # Iterate is safer for reduction
                for cw in list(code_list):
                    # Check if redundant
                    # A word is redundant if removing it leaves no uncovered words
                    # i.e., min(cover_counts[balls[cw]]) > 1
                    # Or simply: if we remove it, are there any 0s in cover_counts[balls[cw]]?
                    
                    # Temporarily remove
                    cover_counts[balls[cw]] -= 1
                    if np.any(cover_counts[balls[cw]] == 0):
                        # Not redundant, revert
                        cover_counts[balls[cw]] += 1
                    else:
                        # Redundant, keep removed
                        code_list.remove(cw)
                        in_code[cw] = False
                        removed = True
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                            best_cover = cover_counts.copy()
                            best_in = in_code.copy()
                            stuck = 0
                        break # Restart scan after removal
            
            if len(code_list) <= 73:
                break
                
            # 2. Swap Move
            # Pick a random word to remove
            cw_out = random.choice(code_list)
            
            # Remove it
            code_list.remove(cw_out)
            in_code[cw_out] = False
            cover_counts[balls[cw_out]] -= 1
            
            # Find uncovered words
            uncovered_mask = (cover_counts == 0)
            
            if not np.any(uncovered_mask):
                # It was redundant (should have been caught above, but safe)
                code_list.append(cw_out)
                in_code[cw_out] = True
                cover_counts[balls[cw_out]] += 1
                continue
            
            # Find best word to add
            # It must cover all uncovered words.
            # Score = number of covered words (or just validity)
            
            best_cw_in = -1
            best_score = -1
            
            # Iterate all candidates
            for cw_in in range(N):
                if in_code[cw_in]: continue
                if cw_in == cw_out: continue
                
                # Does it cover all uncovered?
                # Check if balls[cw_in] contains all indices where uncovered_mask is True
                # sum(uncovered_mask[balls[cw_in]]) == sum(uncovered_mask)
                
                # Optimization:
                # If the number of uncovered words is small, this is fast.
                # If large, it's slow.
                
                # Quick check:
                # The new word must cover the "hard" uncovered words.
                
                # Calculate coverage of uncovered words by cw_in
                cov_uncovered = np.sum(uncovered_mask[balls[cw_in]])
                total_uncovered = np.sum(uncovered_mask)
                
                if cov_uncovered < total_uncovered:
                    continue # Doesn't cover all holes
                
                # If it covers all holes, it's a valid swap.
                # We might prefer one that also covers other already-covered words?
                # No, we just want to maintain validity.
                # To escape local minima, we can accept any valid swap.
                # To improve, we can look for a swap that allows further reduction?
                # For now, accept first valid swap found or random valid.
                
                best_cw_in = cw_in
                break # Take first valid one to save time
                
            if best_cw_in == -1:
                # No valid swap found. Revert.
                code_list.append(cw_out)
                in_code[cw_out] = True
                cover_counts[balls[cw_out]] += 1
                stuck += 1
                
                if stuck > 100:
                    # Perturbation: Add a random word, remove a random word (even if invalid temporarily?)
                    # No, we must maintain validity.
                    # Try a "2-swap": Remove 2, Add 1? Or Remove 1, Add 2?
                    # Remove 1, Add 2 increases size.
                    # Let's try Random Restart if stuck too long.
                    if stuck > 500:
                        return best_code, best_cover, best_in, best_size
            else:
                # Perform swap
                code_list.append(best_cw_in)
                in_code[best_cw_in] = True
                cover_counts[balls[best_cw_in]] += 1
                stuck = 0
                
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                    best_cover = cover_counts.copy()
                    best_in = in_code.copy()
                    
        return best_code, best_cover, best_in, best_size

    # Run Local Search
    code_list, cover_counts, in_code, best_size = try_remove_and_swap(code, cover_counts, in_code, balls, deadline)
    
    # Final Polish
    if time.time() < deadline - 5:
         # Try greedy reduction again on the best found
         # (Already done in loop, but just in case)
         pass
         
    # Convert back to words
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return list(reversed(w))
        
    return [idx_to_word(i) for i in best_code]

def _generic_greedy(q, n, radius):
    # Fallback
    N = q ** n
    if N > 10000:
        raise ValueError("Too large")
        
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx
            
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return list(reversed(w))
        
    all_words = [idx_to_word(i) for i in range(N)]
    
    # Simple Greedy
    uncovered = set(range(N))
    code = []
    
    # Precompute balls
    balls = [None] * N
    for i in range(N):
        w = all_words[i]
        ball = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_w = w[:]
                    new_w[pos] = new_val
                    ball.add(word_to_idx(new_w))
        balls[i] = ball
        
    while uncovered:
        best_cw = -1
        best_gain = -1
        for cw in range(N):
            if cw in code: continue
            gain = len(balls[cw] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cw = cw
        code.append(best_cw)
        uncovered -= balls[best_cw]
        
    return [idx_to_word(i) for i in code]
