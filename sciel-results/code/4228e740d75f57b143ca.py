import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin for 150s limit
    
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
        
    # Precompute powers of q for fast index calculation
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
        
    # Generate neighbor lists (balls) as lists of indices
    # For q=3, n=8, radius=1, each ball has 1 + 8*2 = 17 elements
    balls = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neighbors = [i]
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    idx = i + delta
                    neighbors.append(idx)
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
                        neighbors.append(idx)
        balls[i] = neighbors

    def get_cover_mask(code_indices):
        mask = set()
        for c in code_indices:
            mask.update(balls[c])
        return mask
        
    # Deterministic Greedy with Symmetry Reduction
    # Strategy:
    # 1. Fix c0 = 0 (lex smallest).
    # 2. Use a fast greedy that tracks uncovered count.
    
    def deterministic_greedy():
        uncovered = set(range(total_words))
        code = []
        
        # Step 1: Force c0 = 0
        c0 = 0
        code.append(c0)
        for nb in balls[c0]:
            uncovered.discard(nb)
        
        # Step 2: Greedy selection
        # We need to pick the word that covers the most uncovered words.
        # To speed this up, we can maintain a list of candidate words.
        # However, since the gain changes as we remove words, we must re-evaluate or use a heap.
        # A simple re-evaluation loop is O(N^2) in the number of picks, which is ~400 * 6561 ~ 2.6M ops.
        # This is fast enough in Python if we use sets.
        
        while uncovered:
            if time.time() > deadline - 5.0:
                return code
            
            best_idx = -1
            max_cover = 0
            
            # Iterate through all possible codewords
            # Optimization: We can skip indices already in code? 
            # Membership check in list is O(N). 
            # For speed, we can just calculate gain. 
            # But we must ensure we don't pick a duplicate.
            # We can maintain a set of chosen indices.
            
            # Let's just scan all. 
            for i in range(total_words):
                if i in code:
                    continue
                # Count how many uncovered neighbors i has
                # We can do this by intersecting balls[i] with uncovered
                # len(balls[i] & uncovered)
                # This creates a new set each time. 
                # Alternative: iterate balls[i] and count if in uncovered.
                # Since balls[i] is small (17), this is fast.
                gain = 0
                for nb in balls[i]:
                    if nb in uncovered:
                        gain += 1
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    # Early break? No, we need max.
                
            if max_cover == 0:
                # Should not happen if uncovered is not empty and balls cover self
                break
                
            code.append(best_idx)
            for nb in balls[best_idx]:
                uncovered.discard(nb)
            
        return code

    def local_search(code):
        if code is None:
            return None
            
        current_code = code[:]
        current_size = len(current_code)
        
        # Precompute cover count: for each word, how many codewords cover it?
        # Or simpler: maintain a set of covered words and a count of how many times each is covered.
        # Actually, for removal, we just need to know if removing a codeword leaves any word uncovered.
        # This is equivalent to: for each word covered by c, is it covered by any other codeword?
        
        # Let's use a "coverage count" array.
        # cover_count[i] = number of codewords in current_code that cover i.
        cover_count = [0] * total_words
        for c in current_code:
            for nb in balls[c]:
                cover_count[nb] += 1
        
        improved = True
        # Limit iterations to prevent timeout
        max_iters = 2000 
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Phase 1: Remove redundant
            # Shuffle to avoid bias? Or just iterate.
            # Iterating in reverse allows popping without index shifting issues if we use a list.
            # But we want to try removing ANY redundant word.
            # Let's iterate through a shuffled copy of the code to try removals.
            # This helps escape local minima caused by order.
            order = current_code[:]
            random.shuffle(order)
            
            for c in order:
                if c not in current_code:
                    continue # It might have been removed in a previous step? No, we only remove once per pass.
                
                # Check if c is redundant
                # c is redundant if for all nb in balls[c], cover_count[nb] > 1
                redundant = True
                for nb in balls[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    current_code.remove(c)
                    for nb in balls[c]:
                        cover_count[nb] -= 1
                    current_size -= 1
                    improved = True
                    # Don't break, try to remove more in this pass
                    # But we must be careful: if we remove c, cover_counts change.
                    # The loop continues with the next c in 'order'.
                    # If the next c was dependent on the first c, it might now be non-redundant.
                    # But we check 'cover_count' dynamically, so it's safe.
                    
            if not improved:
                # Phase 2: Swap Search
                # Try to swap a codeword A with B (not in code) such that we can remove C.
                # Or simply: Try to replace A with B if it allows removing C.
                # Heuristic: Pick a random A. Find B that covers the most words uniquely covered by A.
                # Then check if we can remove any C.
                
                if len(current_code) > 1:
                    # Limit attempts
                    for _ in range(20):
                        if time.time() > deadline - 2.0:
                            return current_code
                            
                        # Pick A: random codeword
                        a = random.choice(current_code)
                        
                        # Identify words uniquely covered by A
                        # unique_covered_a = [nb for nb in balls[a] if cover_count[nb] == 1]
                        # We want to find B that covers many of these.
                        
                        # To be efficient, we can just pick a random B not in code?
                        # Or pick B from the neighbors of the uncovered words? But there are no uncovered words.
                        # So we must pick B that "helps" cover words that A covers, so that A can be removed?
                        # No, if we swap A->B, A is gone. So B must cover the words that A was uniquely covering.
                        
                        # Let's find the set of words uniquely covered by A.
                        unique_a = []
                        for nb in balls[a]:
                            if cover_count[nb] == 1:
                                unique_a.append(nb)
                                
                        if not unique_a:
                            # A is not uniquely covering anything. Swapping it doesn't help with removal immediately?
                            # Unless B allows removing C.
                            # But if A is not unique, removing A is already possible (redundant).
                            # So if we are in Phase 2, no codeword is redundant.
                            # Thus every codeword has at least one unique neighbor.
                            pass
                        
                        # Find best B
                        # B should cover as many words in unique_a as possible.
                        # And B should be "good" generally.
                        
                        # Candidate B's: Words that are close to A? Or random?
                        # Let's try words at distance 2 from A?
                        # Or just random sampling from the space.
                        
                        # Let's try a few random candidates.
                        best_b = -1
                        best_gain = 0
                        for _ in range(10):
                            b = random.randint(0, total_words - 1)
                            if b in current_code:
                                continue
                            
                            # Gain: how many words in unique_a does b cover?
                            gain = 0
                            for nb in balls[b]:
                                if nb in unique_a:
                                    gain += 1
                                    
                            if gain > best_gain:
                                best_gain = gain
                                best_b = b
                                
                        if best_b == -1:
                            continue
                            
                        # Tentative swap: Remove A, Add B
                        # Update cover counts
                        for nb in balls[a]:
                            cover_count[nb] -= 1
                        for nb in balls[best_b]:
                            cover_count[nb] += 1
                            
                        # Now check if we can remove any C
                        removed_any = False
                        # We need to check if any C in current_code (except A, which is removed) is now redundant.
                        # Actually, A is removed. So we check all remaining C.
                        # To be fast, check only C that might have become redundant.
                        # C becomes redundant if all its neighbors are covered by others.
                        # The only change is that A's coverage is gone and B's is added.
                        # So C might become redundant if it relied on A? No, if it relied on A, cover_count would be 1 for some nb.
                        # If cover_count[nb] was 1 (covered by A), and A is removed, it becomes 0 (uncovered) unless B covers it.
                        # So we must ensure no word becomes uncovered.
                        
                        # Check validity of the swap
                        valid_swap = True
                        for nb in balls[a]:
                            if cover_count[nb] == 0:
                                valid_swap = False
                                break
                                
                        if not valid_swap:
                            # Revert
                            for nb in balls[a]:
                                cover_count[nb] += 1
                            for nb in balls[best_b]:
                                cover_count[nb] -= 1
                            continue
                            
                        # Swap is valid. Now try to remove a redundant C.
                        # We can remove C if it's redundant in the new configuration.
                        # Let's scan a few candidates for C.
                        # C is likely to be redundant if it was "close" to A or B?
                        # Or just scan all?
                        # Let's scan all, but break early if found.
                        
                        found_c = -1
                        for c in current_code:
                            if c == a:
                                continue
                            # Check if c is redundant
                            redundant = True
                            for nb in balls[c]:
                                if cover_count[nb] <= 1:
                                    redundant = False
                                    break
                            if redundant:
                                found_c = c
                                break
                                
                        if found_c != -1:
                            # Perform removal
                            current_code.remove(found_c)
                            for nb in balls[found_c]:
                                cover_count[nb] -= 1
                            current_size -= 1
                            improved = True
                            # Update current_code to reflect the swap (A removed, B added)
                            # Wait, I haven't updated current_code for the swap yet!
                            # I only updated cover_count.
                            # I need to:
                            # 1. Remove A from current_code
                            # 2. Add B to current_code
                            # 3. Remove C from current_code
                            
                            # Let's do that now.
                            current_code.remove(a)
                            current_code.append(best_b)
                            # found_c was already removed from the list? No, I just checked it.
                            # I need to remove found_c from current_code.
                            current_code.remove(found_c)
                            
                            # The cover_count updates for A and B were already done.
                            # The cover_count update for C was done above.
                            
                            break # Restart main loop
                        else:
                            # No removal found. Revert swap.
                            for nb in balls[a]:
                                cover_count[nb] += 1
                            for nb in balls[best_b]:
                                cover_count[nb] -= 1
                            
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
    # 2. c0=0, c1=lex_largest_dist_2
    # 3. c0=0, c1=random (but deterministic seed)
    
    starts = []
    
    # Generate candidates for c1 (words at distance 2 from 0)
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
        
    for c0, c1 in starts:
        if time.time() > deadline - 20.0:
            break
            
        # Run greedy with these initial picks
        uncovered = set(range(total_words))
        code = []
        
        # Force c0
        code.append(c0)
        for nb in balls[c0]:
            uncovered.discard(nb)
            
        # Force c1
        if c1 not in code:
            code.append(c1)
            for nb in balls[c1]:
                uncovered.discard(nb)
                
        # Continue greedy
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            best_idx = -1
            max_cover = 0
            
            for i in range(total_words):
                if i in code:
                    continue
                gain = 0
                for nb in balls[i]:
                    if nb in uncovered:
                        gain += 1
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                
            if max_cover == 0:
                break
                
            code.append(best_idx)
            for nb in balls[best_idx]:
                uncovered.discard(nb)
                
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
        # Emergency fallback
        return [list(digits_of[i]) for i in range(total_words)]
        
    # Ensure validity: If the code is not valid, we must expand it.
    # Check validity
    cover = set()
    for c in best_candidate:
        cover.update(balls[c])
        
    if len(cover) < total_words:
        # Add uncovered words
        for i in range(total_words):
            if i not in cover:
                best_candidate.append(i)
                cover.update(balls[i])
            if time.time() > deadline - 1.0:
                break

    return [list(digits_of[i]) for i in best_candidate]
