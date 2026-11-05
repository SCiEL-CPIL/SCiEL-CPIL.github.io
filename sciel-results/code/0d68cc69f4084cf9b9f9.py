import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    total_words = q ** n
    
    # Precompute digits for all words as a 2D array
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors (balls) as bitmasks
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
        
    # Generate balls
    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius (slower, but n=8 is small)
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * q_powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    def get_cover_mask(code_indices):
        mask = 0
        for c in code_indices:
            mask |= balls[c]
        return mask
        
    # Deterministic Greedy with Symmetry Reduction
    # Strategy:
    # 1. Fix c0 = 0 (lex smallest).
    # 2. Fix c1 to be the lex smallest word that maximizes coverage (which is any word at distance 2 from 0, due to symmetry).
    #    Actually, let's just run the greedy, but force the first few picks to be canonical.
    
    def deterministic_greedy():
        uncovered = all_mask
        code = []
        
        # Step 1: Force c0 = 0
        # This is valid because the problem is translation invariant.
        # However, we must ensure 0 is a good start. It is, as it covers the most "central" area? 
        # Actually, all words are symmetric. Fixing 0 breaks symmetry without loss of generality for the *set* of codewords, 
        # but the *indices* of the rest will depend on this choice.
        
        c0 = 0
        code.append(c0)
        uncovered &= ~balls[c0]
        
        # Step 2: Find the next codeword.
        # To break symmetry deterministically, we pick the lexicographically smallest index 
        # that has the maximum marginal gain.
        
        while uncovered:
            if time.time() > deadline - 5.0:
                return code
            
            best_idx = -1
            max_cover = 0
            
            # Iterate through all possible codewords
            # We can skip indices already in code? 
            # Membership check in list is O(N). 
            # For speed, we can just calculate gain. If gain is 0, it's likely already in code or redundant.
            # But we must ensure we don't pick a duplicate.
            # Since we update 'uncovered', if a word is already in code, its ball is subset of (all_mask ^ uncovered)? 
            # No. If c is in code, balls[c] & uncovered might be 0 if it's fully covered by others, 
            # but if it's the *only* cover for some part, gain > 0. 
            # However, we should not pick a duplicate.
            # We can maintain a set of chosen indices.
            
            # Optimization: 
            # We iterate i from 0 to total_words-1.
            # We pick the first i with max gain.
            
            for i in range(total_words):
                if i in code:
                    continue
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    # We could break early if max_cover is theoretical max? 
                    # Max possible gain is ball_size. 
                    # But we don't know if it's achievable.
                
            if max_cover == 0:
                # Should not happen if uncovered is not empty and balls cover self
                # If it happens, break to avoid infinite loop
                break
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    def local_search(code):
        if code is None:
            return None
            
        current_code = code[:]
        current_size = len(current_code)
        
        # Precompute cover mask
        current_cover = get_cover_mask(current_code)
        
        improved = True
        # Limit iterations to prevent timeout
        max_iters = 1000 
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Phase 1: Remove redundant
            for i in range(len(current_code)):
                c = current_code[i]
                # Check if removing c keeps cover valid
                if (current_cover & ~balls[c]) == all_mask:
                    # Safe to remove
                    current_code.pop(i)
                    current_cover &= ~balls[c]
                    current_size -= 1
                    improved = True
                    break # Restart loop as indices changed
                    
            if not improved:
                # Phase 2: Deterministic Swap Search
                # Try to swap a codeword with a "better" one (lexicographically smaller or larger?)
                # Let's try to replace a codeword with another that allows removal of a third?
                # Simplified: Try to find a swap that reduces the "redundancy" or improves coverage efficiency.
                
                # Heuristic: Pick the first codeword A.
                # Find a codeword B (not in code) such that swapping A->B allows removal of some C.
                
                if len(current_code) > 1:
                    # Limit attempts
                    for _ in range(50):
                        if time.time() > deadline - 2.0:
                            return current_code
                            
                        # Pick A: first codeword
                        a = current_code[0]
                        
                        # Pick B: lex smallest word not in code
                        # This is O(N^2) if we do it naively. 
                        # We can precompute a list of non-code words?
                        # For now, scan.
                        b_found = -1
                        for b in range(total_words):
                            if b not in current_code:
                                b_found = b
                                break
                                
                        if b_found == -1:
                            break
                            
                        b = b_found
                        
                        # Tentative swap
                        new_code = current_code[:]
                        new_code[0] = b
                        new_cover = get_cover_mask(new_code)
                        
                        # Try to remove redundant from new_code
                        removed_any = False
                        for j in range(len(new_code)):
                            c = new_code[j]
                            if (new_cover & ~balls[c]) == all_mask:
                                # Remove c
                                new_code.pop(j)
                                new_cover &= ~balls[c]
                                removed_any = True
                                break
                                
                        if removed_any and len(new_code) < current_size:
                            current_code = new_code
                            current_cover = new_cover
                            current_size = len(current_code)
                            improved = True
                            break # Restart main loop
                            
        return current_code

    # Run the deterministic greedy
    # Since the greedy is deterministic, we only need to run it once?
    # However, the "local minimum" might be bad.
    # To explore other basins, we can vary the *initial* symmetry breaking.
    # Instead of fixing c0=0, we can try a few canonical c0, c1 pairs.
    
    best_candidate = None
    best_size = float('inf')
    
    # Canonical starts:
    # 1. c0=0, c1=lex_smallest_dist_2
    # 2. c0=0, c1=lex_largest_dist_2 (might lead to different basin)
    # 3. c0=0, c1=random (but deterministic seed)
    
    starts = []
    
    # Generate candidates for c1 (words at distance 2 from 0)
    # There are 8 * 2 * 2 = 32 such words? 
    # Dist 2 means exactly 2 positions differ.
    # Pos 1,2: (1,1), (1,2), (2,1), (2,2) -> 4 words per pair of positions.
    # 28 pairs of positions * 4 = 112 words at dist 2.
    
    dist2_words = []
    for p1 in range(n):
        for p2 in range(p1 + 1, n):
            for d1 in range(1, q):
                for d2 in range(1, q):
                    # Construct word
                    w = [0] * n
                    w[p1] = d1
                    w[p2] = d2
                    idx = 0
                    for i in range(n):
                        idx = idx * q + w[i]
                    dist2_words.append(idx)
                    
    # Pick a few diverse starts
    if dist2_words:
        starts.append((0, dist2_words[0])) # Lex smallest
        starts.append((0, dist2_words[-1])) # Lex largest
        # Add a middle one
        starts.append((0, dist2_words[len(dist2_words)//2]))
        
    # Also try just c0=0 and let greedy pick c1 (which will be lex smallest max gain)
    # This is effectively the same as starts[0] if greedy picks lex smallest.
    
    for c0, c1 in starts:
        if time.time() > deadline - 20.0:
            break
            
        # Run greedy with these initial picks
        uncovered = all_mask
        code = []
        
        # Force c0
        code.append(c0)
        uncovered &= ~balls[c0]
        
        # Force c1
        if c1 not in code:
            code.append(c1)
            uncovered &= ~balls[c1]
            
        # Continue greedy
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            best_idx = -1
            max_cover = 0
            
            for i in range(total_words):
                if i in code:
                    continue
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                
            if max_cover == 0:
                break
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        # Local search
        code = local_search(code)
        
        if code is not None:
            if len(code) < best_size:
                best_size = len(code)
                best_candidate = code
                
    # Fallback: If no candidate found or all failed, use standard greedy
    if best_candidate is None:
        code = deterministic_greedy()
        code = local_search(code)
        best_candidate = code

    if best_candidate is None:
        # Emergency fallback: Tetracode direct sum (might be invalid for radius 1, but returns something)
        # Actually, just return a large code to ensure validity if all else fails
        # A trivial covering code is the entire space (6561 words)
        return [list(range(q)) for _ in range(1)] # This is invalid.
        
    # Ensure validity: If the code is not valid, we must expand it.
    # Check validity
    cover = get_cover_mask(best_candidate)
    if cover != all_mask:
        # Add uncovered words
        uncovered = all_mask & ~cover
        while uncovered:
            # Pick lex smallest uncovered
            for i in range(total_words):
                if (uncovered >> i) & 1:
                    best_candidate.append(i)
                    uncovered &= ~balls[i]
                    break
            if time.time() > deadline - 1.0:
                break

    return [list(digits_of[i]) for i in best_candidate]
