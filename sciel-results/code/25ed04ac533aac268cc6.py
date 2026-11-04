import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Build the direct-sum construction
    # Tetracode is a perfect 1-error-correcting code of length 4 over GF(3)
    # It has 9 words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
            
    m = len(code) # 81 for q=3, n=6
    total_words = q ** n
    
    # Precompute word indices
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword as lists for faster iteration
    covers = []
    for c in code:
        c_set = set()
        c_set.add(word_to_index(c))
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                c_set.add(word_to_index(tuple(c_list)))
            c_list[pos] = orig_val
        covers.append(list(c_set))
    
    # Precompute coverage_count as numpy array for vectorized operations
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1

    best_code = code[:]
    best_count = m
    
    # Precompute: for each codeword, which other codewords cover the same words
    # This helps with swap-based local search
    
    max_restarts = 5000  # Will break based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Reset coverage counts
        curr_coverage = coverage_count.copy()
        active = [True] * m
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy removal pass
        for ci in order:
            if not active[ci]:
                continue
            
            # Check if ci is redundant: all words it covers have coverage > 1
            is_redundant = True
            for wi in covers[ci]:
                if curr_coverage[wi] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                active[ci] = False
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1
        
        # Swap-based local search: try to swap a removed codeword with an active one
        # to see if it enables further removals
        removed = [i for i in range(m) if not active[i]]
        active_list = [i for i in range(m) if active[i]]
        
        improved = True
        while improved:
            improved = False
            for ri in removed:
                for ai in active_list:
                    # Try swapping: remove ai, add ri
                    # First check if adding ri and removing ai maintains coverage
                    # We need to check: after swap, all words still covered
                    # Words only covered by ai (not by ri or others) would be uncovered
                    
                    # Count how many words are covered ONLY by ai among active codewords
                    # and also check if ri covers them
                    ai_only_uncovered = 0
                    for wi in covers[ai]:
                        # Check if wi is covered by any other active codeword besides ai
                        # and if ri covers it
                        if curr_coverage[wi] == 1:
                            # Only ai covers this word
                            # Check if ri covers it
                            if wi not in covers[ri]:
                                ai_only_uncovered += 1
                                break
                    
                    if ai_only_uncovered == 0:
                        # Swap is safe
                        active[ai] = False
                        active[ri] = True
                        for wi in covers[ai]:
                            curr_coverage[wi] -= 1
                        for wi in covers[ri]:
                            curr_coverage[wi] += 1
                        removed.remove(ri)
                        active_list.remove(ai)
                        active_list.append(ri)
                        improved = True
                        break
                if improved:
                    break
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
            if best_count <= 73:
                pass

    # Convert to lists for output
    return [list(c) for c in best_code]
