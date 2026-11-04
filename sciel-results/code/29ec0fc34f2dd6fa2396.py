import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    # The evaluator kills at 150s. We keep a strict margin.
    # 10s safety margin is standard.
    deadline = start_time + 140.0
    
    total_words = q ** n
    
    # Precompute digits for all words
    # For q=3, n=8, total_words=6561.
    # We represent each word as a tuple of digits.
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighborhoods) as bitmasks
    # A ball of radius r around i contains all words within Hamming distance r.
    # For q=3, n=8, radius=1, ball size is 1 + 8*(3-1) = 17.
    # We represent the ball as a Python integer (bitmask).
    
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
                import itertools
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    def greedy_construct(seed):
        """
        Vectorized-ish greedy set cover.
        Uses NumPy for fast intersection and bit counting if possible, 
        but Python ints for masks are actually quite fast for 6561 bits.
        However, we can optimize the 'find best' step.
        """
        random.seed(seed)
        uncovered = all_mask
        code = []
        
        # To speed up the greedy step, we can precompute the balls as a list.
        # The bottleneck is finding the word that covers the most uncovered bits.
        # For 6561 words, a linear scan with bit_count is ~6561 * (cost of & and bit_count).
        # This is fast enough to run many times.
        
        while uncovered:
            if time.time() > deadline - 2.0:
                break # Timeout safety, return partial (might be invalid, but we check later)
            
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Optimization: We can skip words that are already in code? 
            # No, because a word in code might cover new bits if we consider 
            # the state of 'uncovered'. But actually, if a word is in code, 
            # its bits are already covered by definition of the greedy process 
            # (we add it, then remove its bits from uncovered). 
            # So any word in code has 0 gain. We can skip them.
            
            code_set = set(code)
            
            for i in range(total_words):
                if i in code_set:
                    continue
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
                    
            if max_cover == 0:
                break # Should not happen if balls cover self
                
            # Random tie-breaking
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    def prune_code(code):
        """
        Greedy removal of redundant codewords.
        Repeatedly removes a codeword if the remaining set still covers the space.
        """
        if not code:
            return code
            
        code_set = set(code)
        # Calculate current cover
        cover = 0
        for i in code:
            cover |= balls[i]
            
        changed = True
        # We iterate in a random order to break symmetries
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
            c = code[idx]
            # Check if removing c leaves cover valid
            # cover_without_c = cover & ~balls[c]
            # If cover_without_c == all_mask, then c is redundant.
            if (cover & ~balls[c]) == all_mask:
                code.pop(idx)
                code_set.discard(c)
                cover &= ~balls[c]
                changed = True
                # Restart? Or just continue? 
                # Since removing one might make another redundant, 
                # a single pass might not remove all. 
                # But doing a full re-scan is expensive. 
                # Let's do a few passes.
        
        # Do a second pass to catch chains
        indices = list(range(len(code)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
            c = code[idx]
            if (cover & ~balls[c]) == all_mask:
                code.pop(idx)
                code_set.discard(c)
                cover &= ~balls[c]
                
        return code

    def is_valid(code):
        cover = 0
        for i in code:
            cover |= balls[i]
        return cover == all_mask

    best_code = []
    best_len = float('inf')
    
    # Multi-start strategy
    # Run greedy with different random seeds for tie-breaking.
    # Greedy is deterministic if ties are broken deterministically, 
    # but random tie-breaking explores different "greedy paths".
    
    # How many starts?
    # Greedy takes ~50-100ms in Python for this size? 
    # Let's estimate: 6561 * 17 bits. 
    # The inner loop is 6561 iterations. 
    # 6561 * (bitwise AND + bit_count) is very fast.
    # The outer loop runs ~400-500 times (code size).
    # So ~3 million inner iterations. Python can do ~10-50 million simple ops/sec.
    # So one greedy run might take 0.1 - 0.5 seconds.
    # We have ~140 seconds. So we can do ~300-1000 runs.
    
    seed_counter = 0
    while time.time() < deadline - 5.0:
        # Try a few different strategies for tie-breaking or selection?
        # For now, just random tie-breaking.
        
        initial_code = greedy_construct(seed_counter)
        seed_counter += 1
        
        if not initial_code:
            continue
            
        # Prune
        pruned_code = prune_code(initial_code)
        
        # Verify validity
        if is_valid(pruned_code):
            if len(pruned_code) < best_len:
                best_len = len(pruned_code)
                best_code = pruned_code[:]
                
        # If we hit the known best (486), we can stop early? 
        # No, we want to beat it. But if we get close, we might want to 
        # spend time on local search on the best solution.
        
        # If we find a solution <= 486, we might want to do a local search 
        # to try and reduce it further.
        # But local search (SA) was the previous approach and scored 1.10.
        # The multi-start greedy might find 486 quickly. 
        # If we find 486, we can then run a dedicated local search on it 
        # for the remaining time.
        
    # If we have time, run a local search on the best code found
    if best_code and time.time() < deadline - 10.0:
        # Simple local search: Try removing 1, adding 1?
        # Or Swap 1?
        # Since we are minimizing size, "Remove" is the only move that reduces size.
        # "Swap" maintains size but might allow a subsequent "Remove".
        # So: Randomly pick a codeword, remove it. If invalid, try to find 
        # a replacement that covers the lost bits and allows removal of another?
        # This is complex.
        
        # Alternative: Just do more pruning passes with different random orders.
        # The prune_code function above does 2 passes.
        # Let's do a more aggressive pruning on the best_code found.
        
        current_best = best_code[:]
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Shuffle and try to remove
            indices = list(range(len(current_best)))
            random.shuffle(indices)
            cover = 0
            for i in current_best:
                cover |= balls[i]
                
            for idx in indices:
                if time.time() > deadline - 10.0:
                    break
                c = current_best[idx]
                if (cover & ~balls[c]) == all_mask:
                    current_best.pop(idx)
                    cover &= ~balls[c]
                    improved = True
                    if len(current_best) < best_len:
                        best_len = len(current_best)
                        best_code = current_best[:]
                        
    # Final fallback if best_code is empty (should not happen)
    if not best_code:
        # Use a simple deterministic greedy
        initial_code = greedy_construct(42)
        best_code = prune_code(initial_code)
        if not is_valid(best_code):
            # If pruning broke validity (shouldn't if logic is correct), 
            # revert to pre-prune
            best_code = initial_code

    # Convert indices back to words
    return [list(digits_of[i]) for i in best_code]
