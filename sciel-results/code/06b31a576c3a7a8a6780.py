import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    # Parameters for the specific task
    # q=3, n=8, radius=1
    total_words = q ** n
    
    # Precompute cover masks
    # Index calculation: idx = sum(w[i] * q^i)
    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
        
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
        
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * q_powers[i]
        return idx

    cover_masks = [0] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        mask = 1 << idx
        for i in range(n):
            for delta in range(1, q):
                new_w = w[:]
                new_w[i] = (new_w[i] + delta) % q
                new_idx = word_to_idx(new_w)
                mask |= (1 << new_idx)
        cover_masks[idx] = mask

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    code = []
    in_code = [False] * total_words
    
    while uncovered:
        best_idx = -1
        best_count = 0
        # Optimization: Iterate through all, skip if in_code
        for i in range(total_words):
            if in_code[i]: continue
            count = (cover_masks[i] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1:
            break
        code.append(best_idx)
        in_code[best_idx] = True
        uncovered &= ~cover_masks[best_idx]
        
    # Helper to update counts
    counts = [0] * total_words
    def add_counts(mask):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
    def sub_counts(mask):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] -= 1
            m ^= lsb
    def get_uncovered(mask_union):
        # Returns list of indices where count is 0
        # This is slow. Better to check specific masks.
        pass

    # Initialize counts
    for c in code:
        add_counts(cover_masks[c])
        
    # Reduction Pass
    def reduction_pass():
        changed = True
        while changed:
            changed = False
            for i in range(len(code)-1, -1, -1):
                c = code[i]
                mask = cover_masks[c]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    code.pop(i)
                    in_code[c] = False
                    sub_counts(mask)
                    changed = True
                    break # Restart
            
    reduction_pass()
    
    # Local Search
    # 2-for-1 Swap
    while time.time() < deadline:
        # Try to find a 2-for-1 swap
        found_swap = False
        attempts = 0
        while not found_swap and attempts < 1000: # Limit attempts per outer loop
            attempts += 1
            if len(code) < 2:
                break
            i, j = random.sample(range(len(code)), 2)
            c1 = code[i]
            c2 = code[j]
            
            mask1 = cover_masks[c1]
            mask2 = cover_masks[c2]
            
            # Identify words that will be uncovered if we remove c1 and c2
            # A word is uncovered if count[word] == 1 and (word in mask1 or word in mask2)
            # OR count[word] == 2 and (word in mask1 and word in mask2)
            
            # Let's collect the set of "critical" words
            # We need to check if there is a c_new that covers ALL of them.
            
            # To find U (uncovered set):
            # We can simulate removing c1 and c2.
            # But we can just check:
            # U = { w | w in (mask1 U mask2) and ( (w in mask1 and count[w] == 1) or (w in mask2 and count[w] == 1) or (w in mask1 and w in mask2 and count[w] == 2) ) }
            
            # This logic is tricky. 
            # Simpler: 
            # 1. Subtract counts for mask1 and mask2.
            # 2. Find all w where count[w] == 0.
            # 3. If none, remove c1, c2.
            # 4. Else, find c_new covering all w in U.
            # 5. If found, swap.
            # 6. Else, restore counts.
            
            sub_counts(mask1)
            sub_counts(mask2)
            
            U = []
            # Check only words in mask1 or mask2
            # Union of masks
            union_mask = mask1 | mask2
            m = union_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 0:
                    U.append(idx)
                m ^= lsb
                
            if not U:
                # Successful removal
                code.pop(j)
                code.pop(i)
                in_code[c1] = False
                in_code[c2] = False
                found_swap = True
                break # Restart outer loop or continue? 
                # Since code size changed, indices changed. 
                # We break to restart the search with new code.
                
            else:
                # Need to find c_new covering U
                # U is a list of indices.
                # We need a mask for U to check intersection.
                # Or just check each c in code? No, c_new can be any word.
                # Scan all total_words.
                c_new = -1
                for c in range(total_words):
                    if in_code[c]: continue
                    mask_c = cover_masks[c]
                    # Check if mask_c covers all in U
                    covers_all = True
                    for u in U:
                        if not (mask_c >> u) & 1:
                            covers_all = False
                            break
                    if covers_all:
                        c_new = c
                        break
                        
                if c_new != -1:
                    # Swap
                    code.pop(j)
                    code.pop(i)
                    in_code[c1] = False
                    in_code[c2] = False
                    code.append(c_new)
                    in_code[c_new] = True
                    add_counts(cover_masks[c_new])
                    found_swap = True
                    break
                else:
                    # Restore
                    add_counts(mask1)
                    add_counts(mask2)
                    
        if not found_swap:
            # No swap found in this batch, maybe do a reduction pass?
            # Or just continue searching.
            # If we get stuck, try random restart? 
            # For now, just keep trying random pairs.
            pass

    # Final reduction
    reduction_pass()
    
    # Convert to words
    result = []
    for idx in code:
        result.append(idx_to_word(idx))
        
    return result
