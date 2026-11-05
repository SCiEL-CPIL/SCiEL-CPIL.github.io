import itertools
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with radius radius.
    Uses greedy set cover with bitsets and local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Parameters for the specific task: q=3, n=6, radius=1
    # General case handling:
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the seed algorithm for other parameters if needed
        # For this specific task, we assume q=3, n=6, radius=1
        # But to be safe, if it's not the specific case, use a simple heuristic
        # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
        # and "Score: the number of codewords...".
        # The seed used tetracode sum. We will implement the bitset solver for the general case 
        # but it might be too slow for large n. Given the context, it's likely fixed to 3,6,1.
        
        # If not 3,6,1, fall back to seed logic to avoid timeout/errors on unexpected inputs
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy for small spaces
            pass # We assume 3,6,1 for the optimization below

    total_words = q ** n
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute all words
    all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Precompute coverage for each potential codeword
    # coverage[idx] is a bitmask of words covered by codeword idx
    coverages = [0] * total_words
    
    # Generate neighbors for radius 1
    # For each word, generate all words at distance <= 1
    # Distance 0: itself
    # Distance 1: change one position to q-1 other values
    
    # To speed up, we can just iterate all words and check distance, but precomputation is better.
    # For n=6, q=3, total_words=729. 
    # For each of 729 centers, we need to mark 1 + 6*2 = 13 words.
    
    for c_idx, c_word in enumerate(all_words):
        mask = 1 << c_idx
        # Distance 1
        for pos in range(n):
            orig_val = c_word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                # Optimization: calculate index directly
                # idx = c_idx + (new_val - orig_val) * (q ** (n - 1 - pos))
                delta = (new_val - orig_val) * (q ** (n - 1 - pos))
                n_idx = c_idx + delta
                mask |= (1 << n_idx)
        coverages[c_idx] = mask

    all_mask = (1 << total_words) - 1
    
    # Greedy Set Cover
    def greedy_set_cover():
        uncovered = all_mask
        selected = []
        # We need to find the best candidate each time.
        # To speed up, we can just iterate all candidates and find max popcount(coverage & uncovered).
        # 729 candidates, 729 bits. Integer ops are fast.
        
        while uncovered:
            best_c = -1
            best_count = -1
            # Optimization: if we have a time budget, we can break early? No, greedy is deterministic.
            for c in range(total_words):
                if c in selected: # Skip already selected? 
                    # Actually, we can select the same codeword? No, distinct codewords.
                    # So we should skip selected ones.
                    continue
                # Count bits in (coverages[c] & uncovered)
                # Python int.bit_count() is fast
                count = (coverages[c] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
                    if count == 0:
                        break # No more coverage possible, should not happen if valid
            
            if best_c == -1:
                break
            
            selected.append(best_c)
            uncovered &= ~coverages[best_c]
            
            if time.time() > deadline:
                break
                
        return selected

    # Local Search Optimization
    def optimize(selected):
        # selected is a list of indices
        # We want to minimize len(selected)
        
        # Precompute union of coverages for current set
        # This is expensive if done naively each step.
        # Let's maintain the union mask.
        
        def get_union_mask(sel):
            mask = 0
            for idx in sel:
                mask |= coverages[idx]
            return mask
        
        current_sel = list(selected)
        current_union = get_union_mask(current_sel)
        
        if current_union != all_mask:
            return current_sel # Invalid, shouldn't happen from greedy
        
        improved = True
        # Limit iterations to prevent infinite loops
        max_iters = 500
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # 1. Try to remove a codeword
            removed = False
            for i, c in enumerate(list(current_sel)):
                # Check if removing c keeps full coverage
                # new_union = current_union ^ coverages[c] ? No, only if c is unique cover for some bits.
                # Actually, new_union = 0
                # for x in current_sel: if x != c: new_union |= coverages[x]
                # This is O(N^2). N is ~73. 73*73 = 5329 ops. Fast.
                
                # Faster: 
                # deficit = all_mask & ~current_union? No.
                # We want to know if (current_union without c) == all_mask.
                # Let's compute union of others.
                
                # Optimization: 
                # If coverages[c] is a subset of the union of others, we can remove it.
                # union_others = 0
                # for x in current_sel:
                #   if x != c: union_others |= coverages[x]
                # if (union_others & all_mask) == all_mask:
                #    remove c
                
                # Even faster check:
                # Is there any bit in coverages[c] that is NOT in (union of others)?
                # If no, remove.
                
                # Let's just compute union of others for each c.
                union_others = 0
                for x in current_sel:
                    if x != c:
                        union_others |= coverages[x]
                
                if union_others == all_mask:
                    # Remove c
                    current_sel.pop(i)
                    current_union = union_others
                    improved = True
                    removed = True
                    break # Restart loop as set changed
            
            if removed:
                continue
            
            # 2. Try to swap a codeword
            # For each c in current_sel:
            #    Let deficit = all_mask & ~ (union of others)
            #    If deficit is not empty:
            #       Find c_new not in current_sel such that deficit is subset of coverages[c_new]
            #       If found, replace c with c_new.
            
            for i, c in enumerate(list(current_sel)):
                union_others = 0
                for x in current_sel:
                    if x != c:
                        union_others |= coverages[x]
                
                deficit = all_mask & ~union_others
                
                if deficit == 0:
                    # We could have removed it, but maybe we missed it? 
                    # If deficit is 0, it means others cover everything.
                    # This case is handled by step 1.
                    continue
                
                # Find a replacement
                best_new = -1
                # We want a c_new that covers 'deficit'.
                # To be a good swap, it might cover more than just deficit.
                # But for now, just find any valid replacement to change the set.
                # To improve, we might prefer one that allows future removals.
                # Let's just find the first one or the one that covers the most uncovered bits?
                # Since deficit must be covered, any c_new covering deficit is valid.
                
                # Optimization: Iterate through all candidates not in current_sel
                # Check if (coverages[c_new] & deficit) == deficit
                
                # To make it faster, we can precompute for each bit in deficit, which candidates cover it?
                # Or just loop. 729 candidates.
                
                found = False
                for c_new in range(total_words):
                    if c_new in current_sel:
                        continue
                    if (coverages[c_new] & deficit) == deficit:
                        # Valid replacement
                        # Replace
                        current_sel[i] = c_new
                        # Update union
                        current_union = union_others | coverages[c_new]
                        improved = True
                        found = True
                        break
                
                if found:
                    break # Restart loop
            
        return current_sel

    # Run Greedy
    initial_solution = greedy_set_cover()
    
    # Run Optimization
    final_solution = optimize(initial_solution)
    
    # Convert indices back to words
    result = [all_words[idx] for idx in final_solution]
    
    return result
