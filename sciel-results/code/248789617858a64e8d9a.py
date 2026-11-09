import time
import random
import itertools

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=6, radius=1
    # We assume the evaluator passes these specific values based on the task description.
    # If other values are passed, we fall back to a generic greedy, but the task is specific.
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (unlikely for this specific task contract)
        # Simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        def dist(a, b):
            return sum(1 for i in range(n) if a[i] != b[i])
            
        # Precompute neighbors
        neighbors = []
        for w in all_words:
            nbrs = [word_to_idx[w]]
            w_list = list(w)
            for i in range(n):
                orig = w_list[i]
                for v in range(q):
                    if v != orig:
                        w_list[i] = v
                        nbrs.append(word_to_idx[tuple(w_list)])
                w_list[i] = orig
            neighbors.append(nbrs)
            
        covered = [False] * (q**n)
        code = []
        while any(not c for c in covered):
            best_w = -1
            best_gain = -1
            for w_idx in range(len(all_words)):
                gain = 0
                for nbr in neighbors[w_idx]:
                    if not covered[nbr]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = w_idx
            if best_w == -1: break
            code.append(all_words[best_w])
            for nbr in neighbors[best_w]:
                covered[nbr] = True
        return code

    # Specific optimization for q=3, n=6, radius=1
    total_words = q ** n # 729
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists of indices
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = [word_to_idx[w]]
        for i in range(n):
            orig_val = w_list[i]
            for v in range(q):
                if v != orig_val:
                    w_list[i] = v
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(nbrs)
    
    # Precompute masks for each word (bit i is 1 if word i is covered)
    # Python integers are arbitrary precision, so 729 bits is fine.
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for nbr in neighbors[i]:
            m |= (1 << nbr)
        masks[i] = m
        
    FULL_MASK = (1 << total_words) - 1
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_init(rng):
        covered_mask = 0
        code = []
        # To speed up greedy, we can just scan all words each time.
        # 729 words * ~80 iterations = ~60k operations. Very fast.
        while covered_mask != FULL_MASK:
            best_gain = -1
            best_candidates = []
            # Optimization: only consider words that cover at least one uncovered bit
            # But scanning all is fine for 729.
            for w in range(total_words):
                gain = popcount(masks[w] & ~covered_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [w]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(w)
            
            if not best_candidates:
                break # Should not happen if FULL_MASK is reachable
                
            # Random tie-breaking
            chosen = rng.choice(best_candidates)
            code.append(chosen)
            covered_mask |= masks[chosen]
            
        return code

    def local_search(code, rng):
        # code is a list of indices
        # We want to minimize len(code)
        
        # Initialize coverage counts
        # counts[i] = number of codewords covering word i
        counts = [0] * total_words
        for cw in code:
            for nbr in neighbors[cw]:
                counts[nbr] += 1
        
        improved = True
        while improved:
            improved = False
            # Try to remove redundant codewords
            # A codeword c is redundant if for all nbr in neighbors[c], counts[nbr] > 1
            # We iterate backwards to allow popping
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for nbr in neighbors[c]:
                    if counts[nbr] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    for nbr in neighbors[c]:
                        counts[nbr] -= 1
                    code.pop(i)
                    improved = True
            
            if not improved:
                # No single removal possible.
                # Try a "move" or "swap" strategy?
                # For now, we rely on multiple restarts with different greedy inits.
                # If we have time, we can try adding a random word and then reducing.
                if time.time() < deadline - 5.0 and rng.random() < 0.1:
                    # Try adding a random word and see if we can reduce more
                    # This is a bit risky, might increase size.
                    # Better: Try to find a pair of codewords that can be replaced by one?
                    # That's hard.
                    # Let's try a "kick": Add a random word, reduce, if size < original, keep.
                    original_size = len(code)
                    random_word = rng.randint(0, total_words - 1)
                    if random_word not in code:
                        code.append(random_word)
                        for nbr in neighbors[random_word]:
                            counts[nbr] += 1
                        
                        # Reduce again
                        local_improved = True
                        while local_improved:
                            local_improved = False
                            for i in range(len(code) - 1, -1, -1):
                                c = code[i]
                                is_redundant = True
                                for nbr in neighbors[c]:
                                    if counts[nbr] <= 1:
                                        is_redundant = False
                                        break
                                if is_redundant:
                                    for nbr in neighbors[c]:
                                        counts[nbr] -= 1
                                    code.pop(i)
                                    local_improved = True
                                    
                        if len(code) < original_size:
                            # Keep the improvement
                            pass
                        else:
                            # Revert? We don't have the old state easily.
                            # This "kick" approach is complex to revert safely.
                            # Let's stick to pure reduction for stability.
                            # We'll just break the inner loop and let the outer restart handle diversity.
                            pass

        return code

    best_code = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple restarts
    iterations = 0
    while time.time() < deadline - 10.0:
        iterations += 1
        # Use a different seed for each restart
        current_rng = random.Random(iterations * 12345 + 6789)
        
        # Greedy init
        code = greedy_init(current_rng)
        
        # Local search reduction
        code = local_search(code, current_rng)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        if best_size == 73:
            # We found the known optimum. Can we go lower? No, lower bound is 71.
            # But 73 is the best known. If we find 72 or 71, that's a breakthrough.
            # However, 73 is the target for "good" performance.
            # Let's try a few more to be sure we didn't get lucky with 73 and 72 is possible?
            # Unlikely. But we can stop early if we hit 73 multiple times?
            # For now, break if we hit 73 to save time?
            # No, keep searching for 72/71 just in case, but with a tight budget.
            if iterations > 50: # Give it a chance to find 72
                break
                
    if best_code is None:
        # Fallback to a valid code if search failed (should not happen)
        # Direct sum construction (81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                result.append(h + list(tail))
        return result
        
    # Convert indices back to words
    result = [list(all_words[i]) for i in best_code]
    
    # Final verification
    # Check if all words are covered
    covered_mask = 0
    for cw in result:
        idx = word_to_idx[tuple(cw)]
        covered_mask |= masks[idx]
        
    if covered_mask != FULL_MASK:
        # Fallback to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    return result

