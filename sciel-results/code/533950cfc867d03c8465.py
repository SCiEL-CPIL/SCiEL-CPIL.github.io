import time

def construct():
    start_time = time.time()
    best_len = 0
    best_seq = []
    
    # Maximum possible sum: 256*256/2 = 32896
    MAX_SUM = 256 * 256 // 2  # 32896
    # Bitset size: we need to represent sums up to MAX_SUM
    # Python integers can be arbitrarily large, so we use them as bitsets
    
    def backtrack(seq, used_bits, prefix_sums):
        nonlocal best_len, best_seq
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Time guard with safety margin
        if time.time() - start_time > 85:
            return
            
        if not seq:
            last_x = 0
            P_last = 0
        else:
            last_x = seq[-1]
            P_last = prefix_sums[-1]
        
        # Upper bound heuristic: maximum additional elements we can add
        # Given last_x, the smallest possible next value is last_x + 1,
        # then last_x + 2, etc. The maximum number of elements we can add
        # is at most 256 - last_x (since we need strictly increasing values in [1,256])
        max_possible_len = len(seq) + (256 - last_x)
        
        # If even the maximum possible can't beat our best, prune
        if max_possible_len <= best_len:
            return
            
        # Try candidates in ascending order
        for x in range(last_x + 1, 257):
            # Check time guard more frequently
            if time.time() - start_time > 85:
                return
                
            P_new = P_last + x
            
            # New differences: P_new - p for all p in prefix_sums
            # These must all be new (not in used_bits)
            # Use bit operations for O(1) check
            
            # Build the bitmask for new differences
            new_mask = 0
            valid = True
            for p in prefix_sums:
                d = P_new - p
                if d <= 0:
                    valid = False
                    break
                if used_bits & (1 << d):
                    valid = False
                    break
                new_mask |= (1 << d)
                
            if not valid:
                continue
                
            # Also check that new_mask doesn't conflict with itself (all distinct)
            # Actually, since prefix_sums are strictly increasing, P_new - p_i are all distinct
            # because p_i are all different. So no self-conflict.
            
            # Commit
            seq.append(x)
            prefix_sums.append(P_new)
            new_used_bits = used_bits | new_mask
            
            # Recurse
            backtrack(seq, new_used_bits, prefix_sums)
            
            # Undo
            seq.pop()
            prefix_sums.pop()
            
            # Early termination: if we found a solution of length best_len from this branch,
            # and we're just trying to extend, we should continue to try other x values
            # to potentially find longer sequences
            
    backtrack([], 0, [0])
    return best_seq
