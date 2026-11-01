import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        # To get diverse starts, we can randomize the tie-breaking in greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            # Count coverage for each candidate among uncovered words
            # counts[c] = number of uncovered words in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        # We iterate until no more can be removed
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: all words in balls[c] are covered by >= 2 codewords
                # i.e., min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap / Replace
        # Try to replace a codeword c with c' if it leads to a better state
        # Valid swap condition:
        # Let S_c = {x in balls[c] | cover_count[x] == 1} (critical words for c)
        # If we remove c, cover_count[x] becomes 0 for x in S_c.
        # We need c' to cover all x in S_c.
        # i.e., S_c subset of balls[c']
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # First, try to remove (might have become removable due to previous swaps in this pass)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to swap c with some c'
                swapped = False
                # Heuristic: Look for c' that covers the critical set of c
                # Critical set: words covered ONLY by c
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Should have been removed, but just in case
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Try a few random candidates for c'
                # To be faster, we can sample from the entire space or restrict to neighbors
                # Sampling from entire space is O(1) per candidate check if we use sets, but sets are slow.
                # Using boolean arrays is faster.
                
                # For a given c', is critical_words subset of balls[c']?
                # We can check this by: 
                #   is_subset = np.all(np.isin(critical_words, balls[c_prime]))
                # This is slow. 
                # Alternative: 
                #   Create a boolean array for critical_words (size 6561)
                #   Check if any critical_word is NOT in balls[c_prime]
                
                # Optimization: Precompute a "ball membership" matrix? 
                # ball_membership[w] = set of codewords covering w. 
                # But we are modifying cover_count, so we need dynamic checks.
                
                # Let's stick to random sampling but be smart about it.
                # If critical_words is small, checking is fast.
                # If critical_words is large, it's harder to find a c'.
                
                num_trials = 50
                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break
                    c_prime = all_indices[np.random.randint(total)]
                    if c_prime == c:
                        continue
                    
                    # Check if balls[c_prime] covers all critical_words
                    # critical_words are indices.
                    # We need to check if all critical_words are in balls[c_prime].
                    # balls[c_prime] is an array of 17 ints.
                    # We can convert balls[c_prime] to a set for O(1) lookup? 
                    # Creating a set 50 times per codeword per iteration is expensive.
                    
                    # Better: 
                    # Check intersection size.
                    # len(set(critical_words) & set(balls[c_prime])) == len(critical_words)
                    
                    # Even better for small critical_words:
                    # Loop through critical_words and check if any is NOT in balls[c_prime]
                    # Use np.in1d or isin? 
                    # isin is C-implemented and fast for small arrays.
                    
                    valid = True
                    # Critical words are at most 17.
                    for cw in critical_words:
                        if cw not in balls[c_prime]:
                            valid = False
                            break
                            
                    if valid:
                        # Swap is valid
                        cover_count[balls[c]] -= 1
                        cover_count[balls[c_prime]] += 1
                        new_selected.append(c_prime)
                        swapped = True
                        improved = True
                        break
                        # Note: We must NOT revert cover_count if we accept the swap.
                        # But we haven't updated new_selected yet.
                        # We appended c_prime. We skipped c.
                        # The cover_count update is correct.
                    
                if not swapped:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1) - Optional, expensive
        # If len(selected) is still > 450, try this.
        if len(selected) > 450 and time.time() < deadline - 5.0:
            # Try to find two codewords c1, c2 that can be removed and replaced by c3
            # This is complex. We'll stick to single swaps for now as they are more frequent.
            pass
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break # Hit lower bound (unlikely)
            if best_size <= 486:
                # We hit the known best. We can try to improve further or break.
                # Let's keep searching for a bit to see if we can go lower.
                # But 486 is the proven best for a while.
                # Actually, 486 is the BEST KNOWN. If we find 486, we are at the state-of-the-art.
                # We should break to save time for other restarts or just return.
                # But maybe we can find 485? Unlikely.
                # Let's break if we find <= 486 to ensure we return a valid good answer quickly.
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: Return a valid (though large) code
        # Simple greedy without local search
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
