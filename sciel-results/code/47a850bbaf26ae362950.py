import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Generate all words as tuples of integers
    words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as bitsets (Python integers)
    balls = [0] * total_words
    
    for i, w in enumerate(words):
        ball_val = 0
        # Center
        ball_val |= (1 << i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                ball_val |= (1 << j)
        balls[i] = ball_val
        
    best_code = None
    best_size = total_words + 1
    
    start_time = time.time()
    max_time = 110.0 # Leave buffer for safety
    
    while time.time() - start_time < max_time:
        uncovered = (1 << total_words) - 1
        code = []
        code_indices = set()
        
        # Greedy Construction
        while uncovered:
            max_covered = 0
            best_candidates = []
            
            for i in range(total_words):
                if i in code_indices:
                    continue
                covered_count = (balls[i] & uncovered).bit_count()
                
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_candidates = [i]
                elif covered_count == max_covered and covered_count > 0:
                    best_candidates.append(i)
                    
            if max_covered == 0:
                break
                
            chosen_idx = random.choice(best_candidates)
            code.append(words[chosen_idx])
            code_indices.add(chosen_idx)
            uncovered &= ~balls[chosen_idx]
            
        # Local Search
        # We maintain counts of coverage for each word
        counts = [0] * total_words
        for c in code:
            idx = word_to_idx[c]
            b = balls[idx]
            temp = b
            while temp:
                lsb = temp & -temp
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                temp ^= lsb
        
        improved = True
        local_search_iter = 0
        max_local_iter = 500 # Safety cap per restart
        
        while improved and local_search_iter < max_local_iter and time.time() - start_time < max_time:
            improved = False
            local_search_iter += 1
            new_code = []
            new_code_indices = set()
            
            # Pass 1: Check for redundancy and potential swaps
            # We iterate through the current code
            # If we remove/swap, we must update counts.
            # To keep it simple and avoid complex state updates mid-loop, 
            # we will collect a single "best move" (remove or swap) and apply it, then restart the loop.
            
            best_move = None # ('remove', idx) or ('swap', old_idx, new_idx)
            best_move_score = 0 # For swap, maybe prefer swaps that cover more? Or just first found.
            
            current_code_list = code[:]
            current_indices = code_indices.copy()
            
            for c in current_code_list:
                idx = word_to_idx[c]
                b = balls[idx]
                
                # Check if redundant
                is_redundant = True
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        is_redundant = False
                        break
                    temp ^= lsb
                
                if is_redundant:
                    best_move = ('remove', idx)
                    break # Removal is always best
                
                # If not redundant, look for a swap
                # Find unique words covered by this codeword
                unique_mask = 0
                temp = b
                while temp:
                    lsb = temp & -temp
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        unique_mask |= lsb
                    temp ^= lsb
                
                # Find a candidate c' not in code such that unique_mask is subset of balls[c']
                # We can iterate through all words
                # Optimization: Only check words that cover at least one word in unique_mask?
                # For simplicity, check all, but break early if found.
                # To avoid cycling, we might prefer c' with higher "potential" (e.g. covers more total words in current cover?)
                # Or just random choice among valid swaps?
                
                swap_found = False
                valid_swaps = []
                
                # Heuristic: Check candidates that are "close" or just random sampling?
                # 2187 is small. Check all.
                for j in range(total_words):
                    if j in current_indices:
                        continue
                    if (balls[j] & unique_mask) == unique_mask:
                        # Valid swap
                        valid_swaps.append(j)
                
                if valid_swaps:
                    # Choose a swap. 
                    # To improve chances of finding a smaller code later, 
                    # maybe pick the one that maximizes coverage of *other* uncovered parts? 
                    # But everything is covered. 
                    # Maybe pick the one that minimizes "waste" (covers words already covered by many others)?
                    # Or just pick randomly to escape local minima.
                    chosen_swap = random.choice(valid_swaps)
                    best_move = ('swap', idx, chosen_swap)
                    # Don't break immediately? Maybe check for removals first?
                    # We already checked removal for this c. 
                    # We should probably prioritize removals. 
                    # Since we broke out of the inner 'if is_redundant', we are in the swap section.
                    # We can break here to apply the swap and re-evaluate.
                    break
            
            if best_move:
                if best_move[0] == 'remove':
                    idx = best_move[1]
                    # Update counts
                    b = balls[idx]
                    temp = b
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp ^= lsb
                    code.remove(words[idx])
                    code_indices.remove(idx)
                else:
                    old_idx = best_move[1]
                    new_idx = best_move[2]
                    # Update counts for old (decrease)
                    b_old = balls[old_idx]
                    temp = b_old
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp ^= lsb
                    
                    # Update counts for new (increase)
                    b_new = balls[new_idx]
                    temp = b_new
                    while temp:
                        lsb = temp & -temp
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] += 1
                        temp ^= lsb
                        
                    # Update code list and set
                    code.remove(words[old_idx])
                    code_indices.remove(old_idx)
                    code.append(words[new_idx])
                    code_indices.add(new_idx)
                
                improved = True
            else:
                # No move found
                break

        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            if best_size <= 181:
                break
                
    return [list(c) for c in best_code]
