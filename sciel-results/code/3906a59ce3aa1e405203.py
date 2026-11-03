import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks as Python integers (bitsets)
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Greedy Set Cover to get a good initial solution
    def greedy_cover_randomized():
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Simulated Annealing
    def simulated_annealing(initial_code):
        code = initial_code[:]
        code_set = set(code)
        
        # Build cover_count array
        cover_count = [0] * total_space
        for idx in code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        # Check validity (should be valid from greedy)
        # uncovered_mask = 0
        # for i in range(total_space):
        #     if cover_count[i] == 0:
        #         uncovered_mask |= (1 << i)
        
        current_size = len(code)
        best_code = code[:]
        best_size = current_size
        
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.1
        
        # Precompute list of all codeword indices for random selection
        all_indices = list(range(total_space))
        
        while time.time() < deadline - 2.0 and T > min_T:
            # Determine move type
            # 1. Remove (if possible)
            # 2. Swap (neutral)
            # 3. Add (increase size)
            
            move_type = random.random()
            if move_type < 0.3:
                # Try Remove
                # Pick a random codeword
                if not code:
                    T *= cooling_rate
                    continue
                idx_pos = random.randint(0, len(code) - 1)
                idx = code[idx_pos]
                
                # Check if removable: all words it covers must have count > 1
                removable = True
                m = cover_masks[idx]
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    # Perform removal
                    m = cover_masks[idx]
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    
                    code.pop(idx_pos)
                    code_set.remove(idx)
                    current_size -= 1
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = code[:]
                        
            elif move_type < 0.7:
                # Try Swap
                if not code:
                    T *= cooling_rate
                    continue
                idx_pos = random.randint(0, len(code) - 1)
                idx_old = code[idx_pos]
                
                # Pick a random new codeword
                idx_new = random.choice(all_indices)
                if idx_new == idx_old or idx_new in code_set:
                    T *= cooling_rate
                    continue
                    
                # Check if swap is valid:
                # 1. Remove idx_old: find words that would become uncovered (count becomes 0)
                #    These are words where cover_count[w] == 1 and w is in cover_masks[idx_old]
                # 2. Add idx_new: these words must be covered by idx_new
                
                # Find uniquely covered words by idx_old
                unique_mask = 0
                m = cover_masks[idx_old]
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] == 1:
                        unique_mask |= (1 << w_idx)
                    m ^= low
                
                if unique_mask == 0:
                    # idx_old is redundant? If so, we could just remove it.
                    # But we are in Swap mode. If unique_mask is 0, any idx_new works?
                    # Actually, if unique_mask is 0, removing idx_old is valid.
                    # Adding idx_new is always valid.
                    # So this is a valid swap (effectively a removal + addition, but size stays same? No, size stays same).
                    # Wait, if unique_mask is 0, we can just remove idx_old.
                    # But let's treat it as a valid swap to maintain flow.
                    pass 
                else:
                    # Check if idx_new covers all unique words
                    if (unique_mask & ~cover_masks[idx_new]) != 0:
                        # Invalid swap
                        T *= cooling_rate
                        continue
                
                # Perform Swap
                # Remove idx_old
                m = cover_masks[idx_old]
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    cover_count[w_idx] -= 1
                    m ^= low
                
                # Add idx_new
                m = cover_masks[idx_new]
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    cover_count[w_idx] += 1
                    m ^= low
                
                code[idx_pos] = idx_new
                code_set.remove(idx_old)
                code_set.add(idx_new)
                
                # Size unchanged, so no update to best_size unless we found a better structure?
                # No, size is the metric. Swaps don't change size.
                
            else:
                # Try Add
                # Pick a random codeword not in code
                # To avoid infinite loops, we should be careful.
                # If we add, size increases by 1.
                # Acceptance prob: exp(-1/T)
                
                # Find a candidate not in code
                # Simple: pick random, if in code, retry a few times or skip
                cand = random.choice(all_indices)
                if cand in code_set:
                    T *= cooling_rate
                    continue
                
                # Check acceptance
                # Delta E = +1
                if random.random() < math.exp(-1.0 / T):
                    # Add cand
                    m = cover_masks[cand]
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] += 1
                        m ^= low
                    
                    code.append(cand)
                    code_set.add(cand)
                    current_size += 1
                else:
                    # Rejected, state unchanged
                    pass
            
            T *= cooling_rate
            
        # Final Polish: Greedy Removal
        # Try to remove any remaining redundant codewords
        # This is effectively what SA does, but a deterministic pass ensures we don't leave easy wins.
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(len(best_code)):
                idx = best_code[i]
                # Rebuild cover_count for best_code? Or just check on the fly?
                # It's safer to rebuild cover_count for the current best_code candidate
                pass # Skip complex polish if SA is running well. 
                # Actually, let's just return the best found in SA.
                
        return best_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    
    # Run SA
    best_code = simulated_annealing(best_code)
    
    # If time permits, try another greedy + SA restart
    if time.time() < deadline - 30.0:
        cand_code = greedy_cover_randomized()
        cand_code = simulated_annealing(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code

    # Convert indices back to words
    return [all_words[i] for i in best_code]

# EVOLVE-BLOCK-END
