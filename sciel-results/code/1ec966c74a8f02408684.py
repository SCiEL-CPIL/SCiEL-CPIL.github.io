import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100  # Safe margin for 120s limit
    
    # Generate all words and map to indices
    all_words = np.array(list(product(range(q), repeat=n)))
    num_words = q ** n
    
    # Mapping from word tuple to index
    word_to_idx = {}
    for i in range(num_words):
        word_to_idx[tuple(all_words[i])] = i
    
    # Precompute balls as bitsets (Python integers)
    balls = [0] * num_words
    for i in range(num_words):
        w = all_words[i].copy()
        ball = 0
        ball |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                j = word_to_idx[tuple(w)]
                ball |= (1 << j)
                w[pos] = orig
        balls[i] = ball
    
    ALL_WORDS_MASK = (1 << num_words) - 1
    best_code = None
    best_size = num_words
    
    rng = random.Random(42)
    
    # Strategy: Randomized Greedy with Local Search (Redundancy Removal)
    # We run several restarts. Each restart does a greedy build, then local search.
    
    # Determine number of restarts based on time
    # Estimate time per restart: Greedy is O(N * K * bit_ops). Local search is O(K^2 * bit_ops).
    # N=2187, K~200.
    # Let's try 50 restarts first, adjust if time permits.
    
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # 1. Randomized Greedy
        uncovered = ALL_WORDS_MASK
        code = []
        
        # Optimization: Maintain a list of candidates that might still be useful?
        # Simple approach: Iterate all, pick best.
        # To speed up, we can skip candidates that don't cover any new uncovered words.
        
        while uncovered:
            if time.time() > deadline - 20:
                break
            
            best_idx = -1
            best_count = 0
            
            # Randomize tie-breaking slightly by shuffling the iteration order or adding noise
            # Shuffling 2187 items is fast.
            order = list(range(num_words))
            rng.shuffle(order)
            
            for i in order:
                # Quick check: does this ball intersect uncovered?
                # If not, count is 0.
                inter = uncovered & balls[i]
                if inter == 0:
                    continue
                
                # Use bit_count for speed
                cnt = inter.bit_count()
                
                # Randomized tie-breaking: if cnt is close to best, maybe pick this one?
                # Or just strict max with random order handles ties randomly.
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1:
                # Should not happen if uncovered is not 0
                break
            
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
        
        if uncovered != 0:
            continue
        
        current_size = len(code)
        
        # 2. Local Search: Remove redundant codewords
        # Iterate until no change
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            # Shuffle order to remove different ones first
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for idx_pos in indices:
                if time.time() > deadline - 10:
                    break
                
                # Check if code[idx_pos] is redundant
                # Compute union of all other balls
                union_others = 0
                for j, idx in enumerate(code):
                    if j != idx_pos:
                        union_others |= balls[idx]
                
                if union_others == ALL_WORDS_MASK:
                    # Redundant
                    code.pop(idx_pos)
                    changed = True
                    break # Restart the loop with new indices
        
        current_size = len(code)
        
        if current_size < best_size:
            best_size = current_size
            best_code = code.copy()
            
            # Optional: Try simple swaps? (Expensive, skip for now to ensure speed)
            # If we have time, we can try 2-for-1 later.
            
            # Early exit if we hit theoretical lower bound or known best
            if best_size <= 186:
                break
    
    # If we still have time, try to improve the best code found with 2-for-1 swaps
    if best_code is not None and time.time() < deadline - 5:
        code = best_code.copy()
        improved = True
        # Limit iterations for swaps
        max_swap_iters = 10
        for _ in range(max_swap_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            found_swap = False
            # Try removing pairs and replacing with 1
            # This is O(K^2 * N) which is heavy.
            # K~190. 190^2 = 36100 pairs.
            # For each pair, check if union of rest covers all.
            # If not, check if 1 new word covers the gap.
            
            # To make this faster, we only try a subset of pairs or break early.
            # Let's just try to remove single redundant words again (maybe missed due to order)
            # Actually, the previous local search should have caught all single redundancies.
            # 2-for-1 is the next step.
            
            # Given time constraints, let's stick to what we have.
            # The greedy + local search usually gets to ~195-205.
            # Getting to 186 is hard.
            # We will return the best found.
            
            if not improved:
                break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
    # Fallback if no code found (should not happen)
    if best_code is None:
        best_code = list(range(num_words))
        best_size = num_words
        
    # Final Verification
    covered = 0
    for idx in best_code:
        covered |= balls[idx]
    if covered != ALL_WORDS_MASK:
        # Fallback to trivial solution
        best_code = list(range(num_words))
        
    # Convert back to list of lists
    result = [all_words[i].tolist() for i in best_code]
    
    return result
