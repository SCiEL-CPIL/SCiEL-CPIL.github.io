import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (limit is 150s)
    
    total_words = q ** n
    
    # Helper functions
    def int_to_word(idx, q, n):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word

    def word_to_int(word, q):
        val = 0
        for d in word:
            val = val * q + d
        return val

    # Precompute balls as bitmasks
    # balls[i] is a bitmask of all words within distance radius of i
    balls = [0] * total_words
    
    # Generate all words
    all_words = []
    for i in range(total_words):
        all_words.append(int_to_word(i, q, n))
        
    # For each word, compute its ball
    for i, w in enumerate(all_words):
        mask = 0
        # Iterate over all possible words in the ball
        # For radius 1:
        # The word itself
        mask |= (1 << i)
        # Words with one position changed
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                new_idx = word_to_int(new_w, q)
                mask |= (1 << new_idx)
        balls[i] = mask

    # Greedy Set Cover to get initial valid solution
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            best_candidates = []
            
            for c in range(total_words):
                new_covered = balls[c] & uncovered
                count = bin(new_covered).count('1')
                
                if count > best_count:
                    best_count = count
                    best_idx = c
                    best_candidates = [c]
                elif count == best_count and count > 0:
                    best_candidates.append(c)
            
            if best_count == 0:
                break
                
            if len(best_candidates) > 1:
                chosen = random.choice(best_candidates)
            else:
                chosen = best_idx
                
            code_indices.append(chosen)
            uncovered &= ~balls[chosen]
            
            if time.time() > deadline - 5:
                break
                
        return code_indices

    # Simulated Annealing to minimize code size
    def simulated_annealing(initial_code, time_limit):
        code = list(initial_code)
        # Remove duplicates just in case, though greedy shouldn't produce them
        code = list(set(code))
        
        # Calculate current coverage
        covered_mask = 0
        for idx in code:
            covered_mask |= balls[idx]
        
        full_mask = (1 << total_words) - 1
        is_valid = (covered_mask == full_mask)
        
        if not is_valid:
            # Fallback: add missing coverage greedily
            uncovered = full_mask & ~covered_mask
            while uncovered:
                best_idx = -1
                best_count = -1
                for c in range(total_words):
                    new_covered = balls[c] & uncovered
                    count = bin(new_covered).count('1')
                    if count > best_count:
                        best_count = count
                        best_idx = c
                if best_count == 0:
                    break
                code.append(best_idx)
                uncovered &= ~balls[best_idx]
                covered_mask |= balls[best_idx]
                if time.time() > time_limit:
                    break
        
        current_len = len(code)
        best_code = list(code)
        best_len = current_len
        
        # SA Parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < time_limit:
            if T < min_T:
                break
                
            # Generate a neighbor
            # 1. Remove a random codeword if coverage remains valid
            # 2. Swap a random codeword with a random other codeword
            # 3. Add a random codeword (rarely needed if we are minimizing, but helps escape)
            
            move_type = random.random()
            
            new_code = list(code)
            new_covered_mask = covered_mask
            
            if move_type < 0.4 and len(code) > 1:
                # Try removing a random codeword
                idx_to_remove = random.randint(0, len(code) - 1)
                removed_idx = new_code[idx_to_remove]
                
                # Check if any other codeword covers the unique parts of removed_idx?
                # Simpler: just remove and check coverage
                new_code.pop(idx_to_remove)
                
                # Recalculate coverage is expensive. 
                # Optimization: We can check if the removed ball is a subset of the union of others.
                # But for n=6, q=3, 729 words, recalculation is fast enough if we are careful.
                # Let's use a count array approach for speed.
                
                # Actually, let's stick to bitmask for simplicity but optimize the check.
                # We can't easily "subtract" from a union mask.
                # So we must recalculate or maintain counts.
                # Let's maintain a count of how many times each word is covered.
                
                # This requires a different state representation for SA.
                # Let's switch to count-based coverage for SA.
                pass # Placeholder, see below for implementation

            # Due to complexity of bitmask subtraction, we will implement SA using coverage counts.
            # This is more robust for "remove" moves.
            
            break # Exit this dummy loop to use the count-based implementation below

        return best_code

    # Better SA implementation using coverage counts
    def sa_search(initial_code, time_limit):
        code = list(set(initial_code))
        n_code = len(code)
        
        # coverage_counts[w] = number of codewords covering w
        cov_counts = [0] * total_words
        for idx in code:
            mask = balls[idx]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & -m
                bit_index = (lsb.bit_length() - 1)
                cov_counts[bit_index] += 1
                m ^= lsb
        
        # Check validity
        is_valid = all(c > 0 for c in cov_counts)
        if not is_valid:
             # Fix invalidity greedily
             for w in range(total_words):
                 if cov_counts[w] == 0:
                     # Find codeword covering w with max new coverage? Or just any.
                     # Greedy fix
                     best_c = -1
                     best_gain = -1
                     for c in range(total_words):
                         if balls[c] & (1 << w):
                             # Gain is how many 0s it covers
                             m = balls[c]
                             gain = 0
                             mm = m
                             while mm:
                                 lsb = mm & -mm
                                 bi = lsb.bit_length() - 1
                                 if cov_counts[bi] == 0:
                                     gain += 1
                                 mm ^= lsb
                             if gain > best_gain:
                                 best_gain = gain
                                 best_c = c
                     if best_c != -1:
                         code.append(best_c)
                         m = balls[best_c]
                         while m:
                             lsb = m & -m
                             bi = lsb.bit_length() - 1
                             cov_counts[bi] += 1
                             m ^= lsb
                         if time.time() > time_limit:
                             break
        
        current_len = len(code)
        best_code = list(code)
        best_len = current_len
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        while time.time() < time_limit:
            if T < min_T:
                break
                
            # Generate move
            move_type = random.random()
            new_code = list(code)
            new_cov = list(cov_counts) # Copy is expensive, but n=729 is small. 
                                       # Better: modify in place and revert if needed.
            
            # We will modify new_cov in place.
            
            if move_type < 0.5 and len(new_code) > 1:
                # Remove a random codeword
                idx_to_remove = random.randint(0, len(new_code) - 1)
                removed_idx = new_code[idx_to_remove]
                new_code.pop(idx_to_remove)
                
                # Update coverage
                m = balls[removed_idx]
                while m:
                    lsb = m & -m
                    bi = lsb.bit_length() - 1
                    new_cov[bi] -= 1
                    m ^= lsb
                
                # Check validity
                valid = True
                for c in new_cov:
                    if c == 0:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    new_code.insert(idx_to_remove, removed_idx)
                    m = balls[removed_idx]
                    while m:
                        lsb = m & -m
                        bi = lsb.bit_length() - 1
                        new_cov[bi] += 1
                        m ^= lsb
                    # Don't accept this move
                    T *= cooling
                    continue
            else:
                # Swap a random codeword with a random other
                idx_to_swap = random.randint(0, len(new_code) - 1)
                old_idx = new_code[idx_to_swap]
                
                # Choose new_idx from all possible, preferring those not in code?
                # Just pick random from 0..total_words-1
                new_idx = random.randint(0, total_words - 1)
                
                if new_idx == old_idx:
                    T *= cooling
                    continue
                    
                # Check if new_idx is already in code (excluding the one being swapped)
                # If it is, it's a duplicate, which is bad.
                # Simple check:
                is_dup = False
                for i, c in enumerate(new_code):
                    if c == new_idx and i != idx_to_swap:
                        is_dup = True
                        break
                if is_dup:
                    T *= cooling
                    continue

                new_code[idx_to_swap] = new_idx
                
                # Update coverage
                # Remove old
                m = balls[old_idx]
                while m:
                    lsb = m & -m
                    bi = lsb.bit_length() - 1
                    new_cov[bi] -= 1
                    m ^= lsb
                # Add new
                m = balls[new_idx]
                while m:
                    lsb = m & -m
                    bi = lsb.bit_length() - 1
                    new_cov[bi] += 1
                    m ^= lsb
                
                # Check validity
                valid = True
                for c in new_cov:
                    if c == 0:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    new_code[idx_to_swap] = old_idx
                    m = balls[old_idx]
                    while m:
                        lsb = m & -m
                        bi = lsb.bit_length() - 1
                        new_cov[bi] += 1
                        m ^= lsb
                    m = balls[new_idx]
                    while m:
                        lsb = m & -m
                        bi = lsb.bit_length() - 1
                        new_cov[bi] -= 1
                        m ^= lsb
                    T *= cooling
                    continue
            
            # Move is valid
            new_len = len(new_code)
            delta = new_len - current_len
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0:
                accept = True
            else:
                # delta > 0 (worse)
                prob = 2.718281828 ** (-delta / T)
                accept = random.random() < prob
            
            if accept:
                code = new_code
                cov_counts = new_cov
                current_len = new_len
                
                if current_len < best_len:
                    best_len = current_len
                    best_code = list(code)
            
            T *= cooling
        
        return best_code

    # Main Logic
    best_code = None
    best_len = float('inf')
    
    # Get initial greedy solution
    init_code = greedy_cover(random_seed=42)
    
    # Run SA
    # Use a significant portion of time for SA
    sa_time_limit = deadline - 5
    
    final_code = sa_search(init_code, sa_time_limit)
    
    # Verify final code validity one last time
    covered_mask = 0
    for idx in final_code:
        covered_mask |= balls[idx]
    full_mask = (1 << total_words) - 1
    if covered_mask != full_mask:
        # Fallback to greedy if SA broke something (should not happen)
        final_code = init_code
    
    return [int_to_word(idx, q, n) for idx in final_code]
