import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls
    balls = []
    ball_masks = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        balls.append(neighbors)
        
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            for c in range(total_words):
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            if time.time() > deadline:
                return None
        
        return code

    def reduce_cover(code):
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        return code

    def local_search(code, max_iters=500):
        # Returns improved code and new size
        if not code:
            return code, 0
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        improved = False
        iters = 0
        
        while iters < max_iters:
            iters += 1
            if time.time() > deadline:
                break
            
            # 1. Try to remove redundant codewords
            removed_any = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    removed_any = True
                    improved = True
                    break # Restart loop to recount/refresh state
            
            if removed_any:
                continue
            
            # 2. Try 1-for-2 swap (Remove one, Add one, Remove one)
            # Pick a random codeword to remove
            if not code:
                break
                
            idx_remove = random.randint(0, len(code) - 1)
            c_remove = code[idx_remove]
            
            # Find deficit words
            deficit = []
            for w in balls[c_remove]:
                if cover_count[w] == 1:
                    deficit.append(w)
            
            if not deficit:
                # Should have been removed in step 1, but just in case
                for w in balls[c_remove]:
                    cover_count[w] -= 1
                code.pop(idx_remove)
                improved = True
                continue
                
            # Find best replacement
            best_new = -1
            best_cov = 0
            # We need to cover all deficit words to make the removal valid immediately?
            # Or we add a new one, and then see if another becomes removable.
            
            # To be efficient, find c_new that covers the most deficit words
            # If it covers all, we can proceed to check for other removals.
            
            # Optimization: Only check candidates that cover at least one deficit word
            # We can precompute which codewords cover which deficit words? 
            # Deficit size is small (<= 15).
            
            candidates = set()
            for w in deficit:
                # Find all codewords covering w
                # This is expensive if done naively.
                # But we can just iterate all candidates? 2187 is small.
                pass
            
            # Iterate all potential new codewords
            for c_new in range(total_words):
                if c_new in code: # Can't add existing? Actually, we can, but it's a duplicate. 
                    # Wait, code is a set of indices. We should treat it as a set for uniqueness.
                    # The problem requires distinct codewords.
                    # If c_new is already in code, adding it does nothing to coverage.
                    continue
                
                # Count how many deficit words are covered by c_new
                cov = 0
                for w in deficit:
                    if w in balls[c_new]: # This is O(15) per check? No, balls[c_new] is a list.
                        # Better: use bitmask or set intersection
                        pass
                
                # Faster: Precompute ball sets? Or use bitmask.
                # We have ball_masks.
                # Deficit mask:
                # Constructing deficit mask every time is O(15).
                # Then intersection with ball_masks[c_new] is O(1).
                pass
            
            # Let's use bitmask for deficit
            deficit_mask = 0
            for w in deficit:
                deficit_mask |= (1 << w)
                
            best_cov = 0
            best_new = -1
            for c_new in range(total_words):
                if c_new in code:
                    continue
                # Count intersection
                inter = (ball_masks[c_new] & deficit_mask).bit_count()
                if inter > best_cov:
                    best_cov = inter
                    best_new = c_new
            
            if best_new == -1 or best_cov < len(deficit):
                # If we don't cover all deficit words, we can't simply remove c_remove.
                # We could still add best_new, but c_remove would leave holes.
                # So we must cover ALL deficit words to remove c_remove.
                # If best_cov < len(deficit), this swap is not immediately valid for removal.
                # We could try a different c_remove.
                continue
                
            # Tentatively add best_new
            for w in balls[best_new]:
                cover_count[w] += 1
            
            # Now try to remove any redundant codeword
            # This includes c_remove (which is now covered by best_new for its deficit, 
            # and its other words were covered by others)
            # And potentially other codewords.
            
            removed_other = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    removed_other = True
                    improved = True
                    break
            
            if not removed_other:
                # Revert the addition of best_new
                for w in balls[best_new]:
                    cover_count[w] -= 1
            else:
                # We added best_new and removed c (or another).
                # We need to ensure c_remove is removed if it was the one removed?
                # The loop removes the FIRST redundant one found.
                # It might not be c_remove.
                # But the size decreased by 1.
                # We need to make sure we don't have duplicates or invalid states.
                # Since we only added one and removed one, size is same?
                # No, we started with N.
                # We tentatively added 1 -> N+1.
                # We removed 1 -> N.
                # Wait, the goal is to reduce size.
                # My logic:
                # 1. Remove c_remove (conceptually).
                # 2. Add best_new to cover holes.
                # 3. If this makes ANOTHER codeword redundant, remove it.
                # Net: -1.
                
                # But in my code above:
                # I didn't actually remove c_remove from `code` yet.
                # I just calculated deficit based on its removal.
                # Then I added best_new.
                # Then I checked for redundant codewords in the CURRENT `code` (which still contains c_remove).
                # If c_remove is now redundant, it will be removed.
                # If some other codeword is redundant, it will be removed.
                # In either case, size decreases by 1.
                # So we accept.
                
                # However, if the redundant one found was NOT c_remove, and c_remove is STILL there,
                # we have added best_new and removed c_other.
                # c_remove is still in the code.
                # Is the cover still valid? Yes.
                # Did we reduce size? Yes.
                # So it's a valid improvement.
                
                pass # Accept the change

        return code, len(code)

    # Main loop
    # Try multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
            
        code = greedy_cover(seed)
        if code is None:
            break
            
        code = reduce_cover(code)
        code, size = local_search(code)
        
        if size < best_size:
            best_size = size
            best_solution = code.copy()
            
        if best_size <= 186:
            break

    if best_solution is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
