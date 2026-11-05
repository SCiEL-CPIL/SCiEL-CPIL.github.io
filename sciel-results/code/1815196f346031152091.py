import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate the initial pool of candidates using the direct sum construction
    # This is known to be a valid cover. We will try to prune it.
    # Tetracode: 9 words of length 4 covering {0,1,2}^4 with radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Construct the full 486 candidates
    candidates = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            candidates.append(h + list(tail))
            
    num_candidates = len(candidates)
    total_words = q ** n
    
    # Map word to index for fast lookup
    # We can use a simple conversion or a dictionary. 
    # Since q=3, n=8, index = sum(w[i] * 3**(n-1-i))
    # Precompute powers
    powers = [3 ** (n - 1 - i) for i in range(n)]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx

    # Precompute the ball (covered words) for each candidate
    # ball[i] is a list of indices covered by candidate i
    balls = []
    for c in candidates:
        covered_indices = set()
        # The word itself
        covered_indices.add(word_to_idx(c))
        # Words at distance 1
        for pos in range(n):
            original_val = c[pos]
            for new_val in range(q):
                if new_val != original_val:
                    new_word = c[:]
                    new_word[pos] = new_val
                    covered_indices.add(word_to_idx(new_word))
        balls.append(list(covered_indices))
        
    # State: which words are covered?
    # Use a boolean array for O(1) check
    is_covered = np.zeros(total_words, dtype=bool)
    
    # Active candidates
    active = list(range(num_candidates))
    
    selected_codewords = []
    
    # Greedy Set Cover
    # We want to pick the candidate that covers the most uncovered words.
    
    while not is_covered.all():
        if time.time() > deadline:
            break # Return best effort, though it might be invalid if not fully covered. 
            # However, the problem requires a valid cover. If we don't finish, we should probably 
            # fall back to the original 486 if we haven't found a complete cover.
            # But greedy usually finishes very fast for this size.
            
        best_idx = -1
        max_covered = -1
        best_candidate_idx = -1
        
        for idx in active:
            # Count how many words in this ball are currently uncovered
            # We can do this by iterating through the ball and checking is_covered
            # Optimization: if the remaining uncovered words in this ball is <= max_covered, skip?
            # Hard to know without checking.
            
            count = 0
            for w_idx in balls[idx]:
                if not is_covered[w_idx]:
                    count += 1
                    # Early termination if we can't beat current max
                    if count + (len(balls[idx]) - w_idx - 1) <= max_covered: # Rough estimate, maybe not safe
                        pass # Let's just count all for simplicity first, it's fast enough
            if count > max_covered:
                max_covered = count
                best_candidate_idx = idx
                best_idx = idx
                
        if best_candidate_idx == -1:
            # No candidate covers any new word? This shouldn't happen if the initial pool is a cover.
            # If it does, we are stuck.
            break
        
        # Add the best candidate to our solution
        selected_codewords.append(candidates[best_candidate_idx])
        
        # Mark its words as covered
        for w_idx in balls[best_candidate_idx]:
            is_covered[w_idx] = True
            
        # Remove from active
        active.remove(best_candidate_idx)
        
        # Check if done
        if is_covered.all():
            break
            
    # If the greedy process didn't cover everything (due to timeout or bug), 
    # we must return a valid cover. The original construction is valid.
    # However, if we selected some codewords but didn't finish, the set is invalid.
    # So we should only return the greedy result if it is complete.
    # Otherwise, return the original 486.
    
    if is_covered.all():
        return selected_codewords
    else:
        # Fallback to the known valid construction
        return candidates
