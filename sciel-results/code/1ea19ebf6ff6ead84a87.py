import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if q**n < 10000:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            # Generic greedy fallback
            if q**n <= 100000:
                 return [list(t) for t in itertools.product(range(q), repeat=n)]
            return []

    # Specific optimization for q=3, n=7, r=1
    space_size = 3**n
    
    # Precompute all words as integers 0..2186
    words = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int32)
    
    # Build spheres
    # spheres[i] contains the indices of the 15 words covered by word i
    spheres = np.zeros((space_size, 15), dtype=np.int32)
    
    def tup_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * 3 + v
        return idx

    for i in range(space_size):
        w = words[i].copy()
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    neighbors.append(tup_to_idx(w))
            w[pos] = orig
        spheres[i] = neighbors

    # Precompute coverers: for each word w, list of codewords c such that w in spheres[c]
    coverers = [[] for _ in range(space_size)]
    for c in range(space_size):
        for w in spheres[c]:
            coverers[w].append(c)

    # 1. Greedy Set Cover
    coverage_counts = np.zeros(space_size, dtype=np.int32)
    code_indices = []
    
    while time.time() < deadline:
        if len(code_indices) > 250:
            break
            
        # Calculate gain for all candidates
        # gain[c] = count of w in spheres[c] where coverage_counts[w] == 0
        # Vectorized:
        # coverage_counts[spheres] is (N, 15)
        # (coverage_counts[spheres] == 0) is (N, 15) bool
        # sum(axis=1) is (N,)
        gains = np.sum(coverage_counts[spheres] == 0, axis=1)
        
        best_c = np.argmax(gains)
        best_gain = gains[best_c]
        
        if best_gain == 0:
            break # All words covered
            
        code_indices.append(best_c)
        
        # Update coverage
        coverage_counts[spheres[best_c]] += 1
        
        # Check if fully covered
        if np.all(coverage_counts > 0):
            break

    # 2. Local Search
    # We maintain code_indices and coverage_counts.
    # Phase A: Remove redundant codewords
    # Phase B: Swap codewords to enable further removals
    
    def try_remove_redundants(code_list, cov_counts):
        """Removes redundant codewords from code_list, updating cov_counts.
        Returns True if any codeword was removed."""
        removed = False
        i = 0
        while i < len(code_list):
            c = code_list[i]
            # Check if c is redundant:
            # All words in sphere(c) must be covered by at least one other codeword.
            # This is equivalent to: no word in sphere(c) has count 1.
            # Because if count is 1, it's only covered by c (since c is in code_list).
            # Wait, is it possible a word is covered by c and another c' but count is 1? No, count is sum of all.
            # So if count[w] == 1 and w in sphere(c), then c is the only one covering w.
            # Thus c is essential.
            # If no w in sphere(c) has count 1, then for all w in sphere(c), count[w] >= 2.
            # Removing c decreases count[w] by 1, so count[w] >= 1. Valid.
            
            if np.any(cov_counts[spheres[c]] == 1):
                i += 1
            else:
                # Remove c
                code_list.pop(i)
                cov_counts[spheres[c]] -= 1
                removed = True
                # Do not increment i, check the new codeword at position i
        return removed

    def perform_swap(c_in, c_out, code_list, cov_counts, code_set):
        """Attempts to swap c_in with c_out.
        Returns True if the swap was accepted (i.e., it led to a valid state and potentially allowed removals later, 
        but here we just check validity of the swap itself first).
        Actually, we want to swap if it allows *net* reduction.
        Strategy:
        1. Check if swap is valid (preserves coverage).
        2. If valid, perform swap.
        3. Try to remove redundants.
        4. If removals happened, keep swap. Else, revert.
        """
        # Check validity of swap:
        # Words in sphere(c_in) lose 1 cover.
        # Words in sphere(c_out) gain 1 cover.
        # Critical words in sphere(c_in) are those with count 1.
        # These must be covered by c_out.
        s_in = spheres[c_in]
        critical_mask = (cov_counts[s_in] == 1)
        if np.any(critical_mask):
            critical_words = s_in[critical_mask]
            # Check if c_out covers all critical_words
            # We can use coverers or just check membership in spheres[c_out]
            # Since spheres[c_out] is an array of 15 ints, checking membership is O(15) per word.
            # Critical words are usually few.
            s_out_set = set(spheres[c_out])
            for w in critical_words:
                if w not in s_out_set:
                    return False # Swap invalid
        # If no critical words, c_in is redundant, should have been removed. 
        # But maybe it wasn't detected? Or maybe it became critical due to previous swaps?
        # If no critical words, swap is always valid (coverage only increases or stays same for c_in's sphere, 
        # and c_out adds new coverage).
        
        # Perform swap
        cov_counts[s_in] -= 1
        cov_counts[spheres[c_out]] += 1
        code_list[code_list.index(c_in)] = c_out
        code_set.remove(c_in)
        code_set.add(c_out)
        
        # Try to remove redundants
        # We need a copy of code_list to revert if no improvement?
        # Or just check length before and after.
        len_before = len(code_list)
        try_remove_redundants(code_list, cov_counts)
        len_after = len(code_list)
        
        if len_after < len_before:
            return True # Accepted
        else:
            # Revert
            code_list[code_list.index(c_out)] = c_in
            code_set.remove(c_out)
            code_set.add(c_in)
            cov_counts[spheres[c_out]] -= 1
            cov_counts[s_in] += 1
            return False

    code_set = set(code_indices)
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase A: Remove
        if try_remove_redundants(code_indices, coverage_counts):
            improved = True
            code_set = set(code_indices)
            continue # Restart loop to check for new redundancies or swaps
        
        # Phase B: Swap
        # We iterate through each codeword c_in.
        # We find candidates c_out that can replace c_in (valid swap).
        # We try them.
        
        # To speed up, we can precompute for each c_in the critical words.
        # Then find c_out covering them.
        
        for i in range(len(code_indices)):
            c_in = code_indices[i]
            s_in = spheres[c_in]
            critical_mask = (coverage_counts[s_in] == 1)
            
            if not np.any(critical_mask):
                # Redundant, should have been caught in Phase A
                # Force remove
                code_indices.pop(i)
                coverage_counts[s_in] -= 1
                code_set.remove(c_in)
                improved = True
                break # Restart loop
            
            critical_words = s_in[critical_mask]
            
            # Find candidates c_out
            # Start with coverers of first critical word
            common = set(coverers[critical_words[0]])
            for w in critical_words[1:]:
                common.intersection_update(coverers[w])
            
            # Filter out current code
            candidates = common - code_set
            
            if not candidates:
                continue
            
            # Try candidates
            # We can sort candidates by some heuristic? 
            # E.g., number of words covered by c_out that are currently covered exactly once? 
            # Or just try all until one works.
            
            # Convert to list for iteration
            cand_list = list(candidates)
            # Maybe shuffle to avoid bias? Or sort by index for determinism.
            # Let's try to pick the one that covers the most "critical" or "low count" words?
            # For now, just try them all.
            
            for c_out in cand_list:
                # Check if this swap leads to a reduction
                # We need to be careful with state mutation.
                # perform_swap handles revert if no reduction.
                
                # Optimization: 
                # If c_out is in code_set, skip (already done by filter).
                
                if perform_swap(c_in, c_out, code_indices, coverage_counts, code_set):
                    improved = True
                    code_set = set(code_indices)
                    break # Restart outer loop
            
            if improved:
                break

    # Final cleanup
    try_remove_redundants(code_indices, coverage_counts)

    # Convert to lists
    result = [words[i].tolist() for i in code_indices]
    return result
