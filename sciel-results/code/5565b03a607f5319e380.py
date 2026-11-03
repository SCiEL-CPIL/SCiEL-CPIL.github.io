import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    N = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbor masks and lists
    neighbor_masks = [0] * N
    neighbor_lists = [[] for _ in range(N)]
    
    for c_idx in range(N):
        c_word = all_words[c_idx]
        mask = 1 << c_idx
        neighbor_lists[c_idx].append(c_idx)
        
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
        else:
            # General radius (not expected for this task, but good for safety)
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
                        
        neighbor_masks[c_idx] = mask

    def greedy_cover(order=None):
        if order is None:
            order = list(range(N))
            
        uncovered = (1 << N) - 1
        code = []
        
        # We need to pick the best candidate from the remaining ones.
        # If we have a fixed order, we can't just pick the best globally unless we iterate all.
        # Standard greedy: at each step, pick the word that covers the most uncovered.
        # This requires iterating all candidates each step.
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_code = None
            best_count = -1
            
            # Iterate all candidates
            # Optimization: if we have a list of candidates, we can shuffle it?
            # No, greedy requires finding the MAX count.
            
            for c_idx in order:
                # If c_idx is already in code, its contribution is 0?
                # Actually, if c_idx is in code, its bits are cleared.
                # So `uncovered & neighbor_masks[c_idx]` will be 0.
                # So we can just iterate all N.
                
                count = (uncovered & neighbor_masks[c_idx]).bit_count()
                
                if count > best_count:
                    best_count = count
                    best_code = c_idx
                    
            if best_code is None or best_count == 0:
                break
                
            code.append(best_code)
            uncovered &= ~neighbor_masks[best_code]
            
        return code

    def local_search_remove(code):
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                redundant = True
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code

    best_code = []
    
    # Run greedy once
    code = greedy_cover()
    code = local_search_remove(code)
    if len(code) < len(best_code) or not best_code:
        best_code = code
        
    # If time permits, try random restarts
    # Note: greedy_cover with random order doesn't help if we still pick the global max.
    # To use randomization, we need a different strategy, e.g., Randomized Greedy:
    # Pick a random candidate from the top K best?
    # Or just run the same greedy multiple times? It's deterministic.
    # So random restarts of *this* greedy are useless.
    
    # Instead, we can try to improve the current best_code with swaps.
    # Or, we can use a different initialization.
    
    # Given the time, let's just ensure the current result is valid and return it.
    # The greedy + remove is very fast (< 1s).
    # We have ~100s left.
    # We can try a simple local search: 
    # Try to replace each codeword c with a random d.
    # If the new set is valid, and we can remove redundant words, accept.
    
    def try_improve(code):
        code = list(code)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                c = code[i]
                # Try random swaps
                for _ in range(10):
                    d = random.randint(0, N - 1)
                    if d in code:
                        continue
                    
                    # Test replacement
                    test_code = code.copy()
                    test_code[i] = d
                    
                    # Check if valid? 
                    # A quick check is to see if the union of neighbors covers all.
                    # This is expensive.
                    # Instead, check if d covers any word that c covered uniquely?
                    # This is also complex.
                    # Let's just run local_search_remove on the test_code?
                    # That's expensive too.
                    
                    # Alternative: 
                    # Compute cover_counts for current code.
                    # If we replace c with d, 
                    # words covered by c but not d might become uncovered if they were only covered by c.
                    # words covered by d but not c are fine.
                    
                    # Let's compute cover_counts once outside the loop for efficiency?
                    # It changes if we accept a swap.
                    
                    pass
                    
        return code
        
    # Since try_improve is complex, and greedy+remove is strong, 
    # let's just return the best_code found so far.
    
    # Verify validity
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        # Fallback: add missing coverage
        # This should not happen if greedy worked
        uncovered = ((1 << N) - 1) & ~covered_mask
        while uncovered:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
            
    result = [all_words[i] for i in best_code]
    return result
