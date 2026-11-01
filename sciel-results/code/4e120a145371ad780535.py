"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    # Fallback for non-specific cases or if optimization fails
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if q != 3 or n != 7 or radius != 1:
        return fallback()
        
    total_words = q ** n
    # Precompute neighbors for all words
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
        
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbors (distance <= radius)
    # For radius 1, neighbors are the word itself + 1 word change in each of n positions
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[pos] = new_val
                nbs.append(word_to_idx(w_copy))
        neighbors.append(nbs)
        
    neighbors_arr = np.array(neighbors)
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        while uncovered.any():
            # Count how many uncovered words each candidate covers
            # neighbors_arr is (total_words, radius_size)
            # uncovered[neighbors_arr[i]] gives boolean vector of covered status for candidate i
            # sum gives count
            scores = uncovered[neighbors_arr].sum(axis=1)
            
            if randomize:
                # Get max score
                max_score = scores.max()
                # Get indices with max score
                candidates = np.where(scores == max_score)[0]
                # Pick one randomly
                idx = np.random.choice(candidates)
            else:
                idx = np.argmax(scores)
                
            code.append(idx)
            # Update uncovered
            uncovered[neighbors_arr[idx]] = False
            
        return code

    def reduce_code(code):
        code_set = set(code)
        # Count coverage for each word
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
            
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list
            for c in list(code_set):
                # Check if all neighbors of c are covered by at least 2 codewords
                # If min cover count > 1, we can remove c
                min_cov = cover_count[neighbors_arr[c]].min()
                if min_cov > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    changed = True
        return list(code_set)
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Run multiple rounds of greedy + reduction
    # Mix of deterministic and randomized
    for i in range(200):
        if time.time() > deadline:
            break
        if i % 5 == 0:
            cand = greedy_cover(randomize=False)
        else:
            cand = greedy_cover(randomize=True)
            
        cand = reduce_code(cand)
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
            # Optional: If we hit a very good score, we could try local search swaps here
            # But let's keep it simple first
            
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = reduce_code(best_code)
        
    # Convert indices back to words
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
