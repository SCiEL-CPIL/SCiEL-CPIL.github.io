import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        # Tetracode based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    
    # Precompute neighbors (balls) for all 6561 words
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = []
    for w in range(total):
        ball = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                ball.append(new_w)
        balls.append(ball)
    
    balls_np = np.array(balls, dtype=np.int32)
    
    best_size = total
    best_set = None

    # Helper: Remove redundant codewords
    def remove_redundant(codewords):
        if not codewords:
            return []
        cover_count = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_count[balls_np[c]] += 1
            
        redundant_removed = True
        current_set = list(codewords)
        while redundant_removed:
            redundant_removed = False
            # Find words covered by exactly 1 codeword
            unique_words = np.where(cover_count == 1)[0]
            if len(unique_words) == 0:
                # If no unique words, all are redundant? 
                # No, if a word is covered by 0, it's a problem. 
                # But we assume valid input.
                # If cover_count is all >= 1, and no ==1, then all words are covered by >=2.
                # Then we can remove any codeword? 
                # Actually, if a codeword covers only words with count >= 2, it is redundant.
                # So we check each codeword.
                pass
            
            essential_mask = np.zeros(len(current_set), dtype=bool)
            for w in unique_words:
                # Find which codeword in current_set covers w
                # This is slow if done naively.
                # We need a mapping from word to codeword index?
                # Or just check if w is in balls_np[c].
                # Optimization: Precompute for each word, which codewords cover it? 
                # That's 6561 x 6561 matrix? Too big.
                # But we only have ~500 codewords.
                # Let's just iterate.
                for i, c in enumerate(current_set):
                    if w in balls_np[c]:
                        essential_mask[i] = True
                        break
                
            # Remove non-essential
            new_set = []
            for i, c in enumerate(current_set):
                if not essential_mask[i]:
                    # Remove c
                    cover_count[balls_np[c]] -= 1
                    redundant_removed = True
                else:
                    new_set.append(c)
            current_set = new_set
            if not current_set:
                break
                
        return current_set

    # Helper: Local Search
    def local_search(codewords):
        current_set = remove_redundant(codewords)
        if not current_set:
            return []
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            cover_count = np.zeros(total, dtype=np.int32)
            for c in current_set:
                cover_count[balls_np[c]] += 1
            
            # Find a codeword to swap
            # Strategy: Pick a codeword c. 
            # Let U_c be words covered only by c.
            # If U_c is empty, c is redundant (should have been removed).
            # We want to find c' not in set such that ball(c') covers U_c.
            
            # To speed up, we can pick a random codeword from current_set
            # Or pick the one with smallest |U_c|?
            
            random.shuffle(current_set) # For diversity in restarts? No, this is deterministic local search.
            # Let's iterate through all, try to swap one.
            
            for idx, c in enumerate(list(current_set)):
                if time.time() > deadline - 1.0:
                    break
                
                # Compute U_c
                # Words where cover_count == 1 AND covered by c
                # Since cover_count is global, we need to check if c covers it.
                # U_c = { w | cover_count[w] == 1 and w in ball(c) }
                
                # Efficiently:
                # candidates_w = np.where(cover_count == 1)[0]
                # U_c = [w for w in candidates_w if w in balls_np[c]]
                
                # This is still O(N * 17) per codeword.
                # Let's just get unique words globally first.
                unique_words = np.where(cover_count == 1)[0]
                if len(unique_words) == 0:
                    # All words covered by >= 2.
                    # Any codeword is redundant? 
                    # If we remove c, some words might drop to 0? No, they drop to >=1.
                    # So we can just remove one?
                    # remove_redundant should have handled this.
                    # If we are here, it means no codeword is uniquely responsible for any word.
                    # So we can remove any codeword and the set remains valid?
                    # Yes.
                    current_set.pop(idx)
                    improved = True
                    break
                else:
                    # Find U_c
                    U_c = []
                    for w in unique_words:
                        if w in balls_np[c]:
                            U_c.append(w)
                    
                    if len(U_c) == 0:
                        # c is redundant
                        current_set.pop(idx)
                        improved = True
                        break
                    
                    # Try to find c' not in current_set that covers U_c
                    # c' must cover all w in U_c.
                    # We can sample candidates or iterate.
                    # Iterating 6561 is fast (~1ms).
                    
                    found = -1
                    # Optimize: Check if c' covers U_c.
                    # Since U_c is small, we can check intersection.
                    # Precompute set of U_c for O(1) lookup?
                    U_c_set = set(U_c)
                    
                    # To make it faster, we can prioritize c' that covers more "unique" words?
                    # No, just find any valid one.
                    
                    # Optimization: 
                    # We know c' must be in the intersection of balls of all w in U_c.
                    # Intersection of balls is small.
                    # Let's compute intersection.
                    candidates = set(range(total))
                    for w in U_c:
                        # candidates = candidates.intersection(set(balls_np[w]))
                        # This is expensive.
                        pass
                    
                    # Simpler: Iterate through all c' not in current_set.
                    # current_set is ~500.
                    # So we check ~6000 candidates.
                    # For each, check if it covers U_c.
                    # Check: all(w in balls_np[c'] for w in U_c)
                    # Since |U_c| is small, this is fast.
                    
                    for c_prime in range(total):
                        if c_prime in current_set:
                            continue
                        # Check if c_prime covers all in U_c
                        valid = True
                        for w in U_c:
                            if w not in balls_np[c_prime]:
                                valid = False
                                break
                        if valid:
                            found = c_prime
                            break
                    
                    if found != -1:
                        # Perform swap
                        current_set[idx] = found
                        # Run remove_redundant
                        current_set = remove_redundant(current_set)
                        improved = True
                        break # Restart the outer loop with new set
            
        return current_set

    # Run multiple restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Greedy Initialization
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break
            
            counts = uncovered_mask[balls_np].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls_np[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue
            
        # Local Search
        refined_set = local_search(selected)
        
        if len(refined_set) < best_size:
            best_size = len(refined_set)
            best_set = refined_set
            
    # Fallback
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]
        
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
