import itertools
import time
import sys

def construct(q, n, radius):
    # For this specific task: q=3, n=7, radius=1
    # But the function signature is general. We will specialize for the known task 
    # but keep the structure. If parameters differ, we might fall back to a simple 
    # construction or raise an error, but the prompt implies this is for the specific 
    # ternary covering code problem.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction if parameters change, 
        # though the task is specific.
        # Direct sum of tetracode is valid for q=3.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all words (valid but large)
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific optimization for 3, 7, 1
    total_words = q ** n
    all_words = [list(word) for word in itertools.product(range(q), repeat=n)]
    
    # Precompute masks
    # mask[i] is an integer with bits set for all words within radius 1 of all_words[i]
    masks = [0] * total_words
    
    # Helper to get index from word
    def word_to_idx(word):
        idx = 0
        for digit in word:
            idx = idx * q + digit
        return idx

    # Precompute neighbors for each word
    for i in range(total_words):
        w = all_words[i]
        mask = 1 << i  # itself
        # Neighbors at distance 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                j = word_to_idx(w)
                mask |= (1 << j)
            w[pos] = orig
        masks[i] = mask

    full_mask = (1 << total_words) - 1
    
    # Greedy Initialization
    covered = 0
    C = [] # list of indices
    
    # Optimization: Keep a list of remaining words to consider?
    # Or just iterate all each time. 2187 is small.
    while covered != full_mask:
        best_idx = -1
        best_gain = -1
        for i in range(total_words):
            if i in C: # This check is O(|C|). 
                continue
            # We can skip if i is already covered? No, we need to cover *new* words.
            # Gain is popcount(masks[i] & ~covered)
            new_cover = masks[i] & ~covered
            gain = bin(new_cover).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        C.append(best_idx)
        covered |= masks[best_idx]
        
    # Convert to set for faster lookup
    C_set = set(C)
    
    # Local Search
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Redundancy Removal
        # We need to check if any c in C is redundant.
        # Redundant if mask[c] is subset of OR(masks[c'] for c' in C, c' != c)
        
        # To make this efficient, we can iterate and try to remove.
        # We'll do a pass.
        to_remove = []
        for c in C:
            # Calculate OR of all other masks
            rest_cover = 0
            for c2 in C:
                if c2 != c:
                    rest_cover |= masks[c2]
            if masks[c] & ~rest_cover == 0:
                to_remove.append(c)
        
        if to_remove:
            for c in to_remove:
                C.remove(c)
                C_set.remove(c)
                covered &= ~masks[c] # This is not strictly correct for 'covered' variable 
                                     # if we use it later, but we recalculate or don't use it 
                                     # for redundancy check directly, we use rest_cover.
                                     # However, 'covered' is used in greedy. 
                                     # In local search, we rely on C_set and masks.
            improved = True
            continue # Restart loop as C changed significantly
        
        # 2. 2-for-1 Replacement
        # Try to replace two codewords with one.
        # Find c1, c2 in C such that Exclusive(c1, c2) is covered by some c_new not in C.
        
        # To speed up, we can limit the number of pairs checked or use heuristics.
        # But 200^2 is 40k pairs. 40k * 2187 checks is ~87M ops. 
        # In Python, 87M simple ops might take ~10-20s. 
        # We have ~100s left. So we can do a few iterations.
        
        found_replacement = False
        # Iterate over pairs
        # We can break early if found
        for i in range(len(C)):
            c1 = C[i]
            for j in range(i + 1, len(C)):
                c2 = C[j]
                
                # Calculate Exclusive coverage of c1 and c2
                # Exclusive = (mask[c1] | mask[c2]) & ~(OR of all other masks)
                
                # Efficient way to get OR of all others:
                # We don't have a global OR of others precomputed.
                # We can compute it on the fly? 
                # OR of C \ {c1, c2}.
                
                # Since we just did redundancy removal, no single codeword is redundant.
                # But the pair might be.
                
                # Let's compute rest_cover for this pair.
                rest_cover = 0
                for c3 in C:
                    if c3 != c1 and c3 != c2:
                        rest_cover |= masks[c3]
                
                exclusive = (masks[c1] | masks[c2]) & ~rest_cover
                
                if exclusive == 0:
                    # This implies c1 and c2 are both redundant relative to others?
                    # Or their union is covered.
                    # If exclusive is 0, we can just remove one of them?
                    # Actually, if exclusive is 0, it means every word covered by c1 or c2 is covered by others.
                    # So we can remove c1 AND c2.
                    C.remove(c1)
                    C_set.remove(c1)
                    C.remove(c2)
                    C_set.remove(c2)
                    improved = True
                    found_replacement = True
                    break
                else:
                    # Check if there is a c_new not in C that covers 'exclusive'
                    # exclusive & ~masks[c_new] == 0
                    
                    # Optimization: 'exclusive' has few bits.
                    # We can iterate through all possible c_new.
                    for c_new in range(total_words):
                        if c_new in C_set:
                            continue
                        if exclusive & ~masks[c_new] == 0:
                            # Replace c1, c2 with c_new
                            C.remove(c1)
                            C_set.remove(c1)
                            C.remove(c2)
                            C_set.remove(c2)
                            C.append(c_new)
                            C_set.add(c_new)
                            improved = True
                            found_replacement = True
                            break
                    if found_replacement:
                        break
            if found_replacement:
                break
        
        if found_replacement:
            continue # Restart loop
        
        # If no 2-for-1 found, try 1-for-1 swap?
        # 1-for-1 doesn't reduce size, but might allow future 2-for-1.
        # Given time, let's stick to 2-for-1 and redundancy.
        
        if not improved:
            break
            
    # Convert indices back to words
    return [all_words[i] for i in C]
