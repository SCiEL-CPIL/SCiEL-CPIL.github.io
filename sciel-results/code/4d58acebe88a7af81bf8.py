import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
        pass

    total_words = q ** n
    # Map word tuple to int index
    # 0..2186
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        # Convert i to base-q representation
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w = tuple(w)
        
        # Neighbors: distance 0 and 1
        # Distance 0
        neighbors[i].append(i)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                # Convert back to int
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                neighbors[i].append(idx)
                
    # Greedy Construction
    def greedy_construct():
        covered = np.zeros(total_words, dtype=bool)
        code = []
        remaining = np.ones(total_words, dtype=bool)
        
        while np.any(remaining):
            # Find word that covers most remaining
            # For each candidate c, count how many neighbors are in remaining
            # This is slow if done naively for every step.
            # Optimization: Only check candidates that are not yet in code? 
            # Actually, we can just check all, or maintain a priority queue.
            # With 2187 words, checking all 2187 candidates each step is ~ 2187 * 2187 * 15 ops?
            # No, for each candidate, we sum `remaining[neighbors[c]]`.
            # 2187 candidates * ~15 neighbors = ~30k ops per step.
            # Max steps ~ 200. Total ~ 6M ops. Fast enough in Python? Maybe.
            
            best_c = -1
            max_covered = -1
            # To speed up, we can sample or use a heuristic.
            # But let's try full search first.
            
            # Vectorized approach might be tricky with dynamic 'remaining'.
            # Let's stick to Python loop but optimized.
            
            # Optimization: Precompute neighbor lists as numpy arrays?
            # neighbors_np = [np.array(nbrs) for nbrs in neighbors]
            
            for c in range(total_words):
                # Count how many neighbors of c are in remaining
                # sum(remaining[nbr] for nbr in neighbors[c])
                # Using numpy:
                # cnt = np.sum(remaining[neighbors_np[c]])
                # This creates an array view and sums.
                pass # Placeholder for logic below

            # Let's use a simpler heuristic for the first pass to get a valid code fast,
            # then improve.
            # Actually, the direct sum (243) is a very good starting point.
            # Let's start from the direct sum code and try to improve it via local search.
            # This is much faster than greedy from scratch.
            
            return code # Will be replaced by start-from-direct-sum logic

    # Start from the known 243-word code (Direct Sum)
    # Tetracode: 9 words of length 4
    # Tails: 27 words of length 3
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a+b)%3, (a+2*b)%3))
            
    initial_code = []
    for h in tetracode:
        for tail in product(range(3), repeat=3):
            initial_code.append(h + tail)
            
    # Convert to indices
    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * 3 + v
        return idx

    code = [tuple_to_idx(w) for w in initial_code]
    
    # Maintain count array
    # count[i] = number of codewords in code that cover i
    count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        count[neighbors[c]] += 1
        
    def is_valid(count_arr):
        return np.all(count_arr > 0)
        
    def try_remove(idx_in_code):
        c = code[idx_in_code]
        # Check if removing c makes any count 0
        # We need to check if min(count[neighbors[c]]) > 1
        # If min > 1, safe to remove.
        # If min == 1, removing it will uncover some words.
        min_cov = np.min(count[neighbors[c]])
        if min_cov > 1:
            # Remove
            count[neighbors[c]] -= 1
            code.pop(idx_in_code)
            return True
        return False

    def try_swap(idx_in_code, new_c):
        old_c = code[idx_in_code]
        if old_c == new_c:
            return False
            
        # Check if swap is beneficial or at least valid
        # We want to see if the new set is valid.
        # New count = count - neighbors[old_c] + neighbors[new_c]
        # We need new_count > 0 everywhere.
        
        # Optimization: Only swap if it covers more "tight" words or something?
        # Simple strategy: Accept if valid.
        
        # Efficient check:
        # Words that become uncovered are those where count[nbr] == 1 and nbr is in neighbors[old_c] but not in neighbors[new_c] (or count decreases to 0).
        # Actually, just compute the delta.
        
        # Let's just do the full check for simplicity, it's fast enough for ~200 words.
        # But we do many swaps.
        
        # Check validity of (code - {old_c} + {new_c})
        # Words at risk: neighbors[old_c]
        # For w in neighbors[old_c]:
        #   new_count_w = count[w] - 1 + (1 if w in neighbors[new_c] else 0)
        #   if new_count_w == 0: invalid
        
        # We can precompute a set of neighbors for faster lookup?
        # Or just use the array.
        
        # Let's try a random swap and check validity.
        
        # To make it faster, we only propose swaps that cover at least one word that is currently covered by exactly 1 codeword (the one being removed)?
        # No, let's just do random restarts of local search.
        
        pass

    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Remove redundant words
        i = 0
        while i < len(code):
            if try_remove(i):
                improved = True
                # Do not increment i, check the new word at i
            else:
                i += 1
                
        # 2. Swap
        # Randomly pick a word to swap out
        # And a random word to swap in
        # If valid, accept.
        
        # To improve, we can try to swap a word that has low "redundancy" with a word that covers more "scarce" words.
        # But simple random swap + validity check works.
        
        # We need to be careful: if we swap and it's valid, we accept.
        # This might increase the number of codewords? No, swap keeps size same.
        # But it might allow further removals.
        
        # Let's do a few random swaps per iteration.
        for _ in range(100):
            if time.time() > deadline:
                break
            if not code: break
            i = random.randint(0, len(code)-1)
            old_c = code[i]
            new_c = random.randint(0, total_words - 1)
            
            if old_c == new_c:
                continue
                
            # Check if swap is valid
            # We need to verify that all words are still covered.
            # Only words in neighbors[old_c] are affected.
            
            # Optimization:
            # Let's compute the change in count for neighbors[old_c].
            # If any word drops to 0, invalid.
            
            valid = True
            # We can use numpy for this check
            # affected = neighbors[old_c]
            # new_counts = count[affected] - 1
            # # Add back if in neighbors[new_c]
            # # This is tricky with sets.
            
            # Let's just do the full update and check, then rollback if invalid.
            # Since we are in a tight loop, rollback is expensive.
            
            # Better: Check first.
            # For w in neighbors[old_c]:
            #   if count[w] == 1:
            #       # This word is covered ONLY by old_c
            #       # So new_c MUST cover w
            #       if w not in neighbors[new_c]: # This 'in' check is slow
            #           valid = False
            #           break
            # If valid so far, we also need to ensure we don't break anything else?
            # No, if count[w] > 1, removing old_c leaves count[w]-1 >= 1.
            # Adding new_c only increases counts.
            # So the only risk is words with count==1 in neighbors[old_c].
            
            # So:
            # risky_words = [w for w in neighbors[old_c] if count[w] == 1]
            # If all risky_words are in neighbors[new_c], the swap is valid.
            
            risky_words = neighbors[old_c]
            # Filter risky words
            # We can use numpy:
            # risky_mask = (count[risky_words] == 1)
            # risky_indices = risky_words[risky_mask]
            
            # Check if risky_indices subset of neighbors[new_c]
            # We can convert neighbors[new_c] to a set? No, too slow to build.
            # We can use a boolean array `in_new_c`?
            
            # Let's maintain a global `neighbor_set`? No.
            
            # Alternative:
            # Precompute `is_neighbor[new_c][w]`? Too much memory (2187x2187).
            
            # Let's just iterate.
            valid = True
            for w in neighbors[old_c]:
                if count[w] == 1:
                    # Must be covered by new_c
                    # Check if w is in neighbors[new_c]
                    # We can do this by checking if any neighbor of new_c is w?
                    # Or just:
                    if not any(x == w for x in neighbors[new_c]):
                        valid = False
                        break
            
            if valid:
                # Perform swap
                count[neighbors[old_c]] -= 1
                count[neighbors[new_c]] += 1
                code[i] = new_c
                improved = True

    # Final cleanup: remove any redundant words again
    i = 0
    while i < len(code):
        if try_remove(i):
            improved = True
        else:
            i += 1

    # Convert back to list of lists
    def idx_to_tuple(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return list(reversed(w))
        
    return [idx_to_tuple(c) for c in code]

from itertools import product
