import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task is specific
        # For now, return a trivial solution or empty
        return []

    # Total words in space
    total_words = q ** n
    all_ones = (1 << total_words) - 1
    
    # Generate all words
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(w)
        
    # Precompute balls: ball[i] is a bitmask of words within distance 1 of words[i]
    balls = [0] * total_words
    for i in range(total_words):
        mask = 1 << i
        # Distance 0
        # Distance 1: change one coordinate
        w = words[i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Construct new word index
                # We can precompute indices or calculate on fly
                # Calculating index: sum(w[j] * q^j)
                # Since n=6, q=3, this is fast
                new_w = w[:]
                new_w[pos] = new_val
                idx = 0
                power = 1
                for j in range(n):
                    idx += new_w[j] * power
                    power *= q
                mask |= (1 << idx)
        balls[i] = mask

    # Greedy Set Cover
    covered = 0
    code_indices = []
    remaining = all_ones
    
    # We need to find the word that covers the most uncovered bits
    # We can optimize by maintaining a list of candidates, but 729 is small.
    
    while remaining != 0:
        best_gain = -1
        best_idx = -1
        # Iterate through all words to find the one with max new coverage
        # Optimization: only check words that haven't been added? 
        # Actually, re-adding is useless, but checking all is simpler and 729 is small.
        for i in range(total_words):
            if i in code_indices: # Skip if already added? 
                # We can just check gain. If already added, gain is 0 for its unique part?
                # No, it might cover new bits if combined? No, balls are fixed.
                # If i is in code, its ball is already in 'covered'. 
                # So balls[i] & ~covered is 0.
                # So we can just check all i.
                pass
            
            gain = (balls[i] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        covered |= balls[best_idx]
        remaining = ~covered & all_ones
        
        if time.time() > deadline - 10:
            break

    # Local Search
    # We represent the code as a set of indices for O(1) lookup
    code_set = set(code_indices)
    
    def is_covered(code_set):
        c = 0
        for idx in code_set:
            c |= balls[idx]
        return c == all_ones

    # Refinement Loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try Deletion
        for idx in list(code_set):
            new_set = code_set.copy()
            new_set.remove(idx)
            if is_covered(new_set):
                code_set = new_set
                improved = True
                break # Restart deletion loop with new set
                
        if time.time() > deadline - 5:
            break
            
        # 2. Try Swap
        # For each c in code, try replacing with w not in code
        # Optimization: Only try swaps if we think they might help?
        # Just iterate.
        for c in list(code_set):
            # Calculate coverage without c
            cov_without_c = 0
            for idx in code_set:
                if idx != c:
                    cov_without_c |= balls[idx]
            
            # We need balls[w] to cover the gap: all_ones & ~cov_without_c
            # Actually, we need (cov_without_c | balls[w]) == all_ones
            # Which means balls[w] must cover (all_ones & ~cov_without_c)
            gap = all_ones & ~cov_without_c
            
            # If gap is 0, we could just delete c (handled above)
            if gap == 0:
                continue
                
            # Find a w not in code_set such that balls[w] covers gap
            # And ideally, w adds more coverage elsewhere? 
            # For a valid swap, we just need the union to be full.
            # Since we are minimizing size, a swap doesn't change size.
            # But it might enable future deletions.
            
            # To speed up, we can just check a random subset of w's or all w's
            # 729 checks per c. 75 * 729 = ~55k checks.
            # Each check: (cov_without_c | balls[w]) == all_ones
            # This is fast.
            
            for w in range(total_words):
                if w in code_set:
                    continue
                if (cov_without_c | balls[w]) == all_ones:
                    # Perform swap
                    code_set.remove(c)
                    code_set.add(w)
                    improved = True
                    break # Restart swap loop with new set
            
            if improved:
                break
        
    # Convert indices back to words
    result = [words[i] for i in code_set]
    
    # Final sanity check
    if not is_covered(code_set):
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return result
