import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 128
    
    # Precompute bit masks for sums?
    # PairSumInS[a] is a bitmask of b such that a+b is in S.
    
    def update_pair_sums(s_mask, x, added):
        """
        Update PairSumInS when x is added or removed from s_mask.
        added: True if x was added, False if removed.
        Returns updated PairSumInS.
        """
        # We need to modify PairSumInS in place or return new.
        # Let's pass it and modify.
        pass # Implemented in main loop for speed

    # We will maintain s_mask and PairSumInS list.
    
    def is_valid_addition(s_mask, x, pair_sums):
        """
        Check if adding x is valid.
        pair_sums[a] is bitmask of b such that a+b in s_mask.
        """
        # A = {a in S | a+x in S}
        # a+x in S means bit (a+x) in s_mask is 1.
        # So a must be in S and a+x in S.
        
        # Construct A_mask
        A_mask = 0
        # Iterate a in S
        # We can just iterate a from 1 to N
        # If a in S and a+x in S, set bit a in A_mask.
        
        # Optimization: 
        # a+x in S => a in S and (s_mask >> (a+x)) & 1.
        
        # Let's just loop a
        for a in range(1, N+1):
            if (s_mask >> a) & 1:
                if a + x > N:
                    continue
                if (s_mask >> (a+x)) & 1:
                    A_mask |= (1 << a)
        
        # Now check if there exist a, b in A_mask, a != b, such that a+b in S.
        # a+b in S means bit b in pair_sums[a] is 1.
        # So we need pair_sums[a] & (A_mask ^ (1<<a)) != 0.
        
        m = A_mask
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            
            # Check if any b in A_mask, b != a, has a+b in S
            # candidates_b = A_mask & ~(1 << a)
            # if pair_sums[a] & candidates_b: return False
            
            candidates_b = A_mask & ~(1 << a)
            if pair_sums[a] & candidates_b:
                return False
        
        return True

    def count_bits(mask):
        return bin(mask).count('1')

    best_mask = 0
    best_size = 0
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        if attempt == 0:
            candidates = list(range(N, 0, -1))
        else:
            candidates = list(range(1, N+1))
            random.shuffle(candidates)
        
        s_mask = 0
        pair_sums = [0] * (N + 1)
        
        for x in candidates:
            if is_valid_addition(s_mask, x, pair_sums):
                # Add x
                s_mask |= (1 << x)
                # Update pair_sums
                # For a < x, if x-a in S, set bit (x-a) in pair_sums[a]
                # Note: x-a must be in S (old s_mask before adding x? No, x is being added, so x-a is in S if it was there).
                # Actually, if we add x, new sums involving x are x.
                # So a+b = x.
                # For a in 1..x-1, b = x-a.
                # If b in S (current s_mask before adding x? No, s_mask is being updated).
                # Let's update based on s_mask BEFORE adding x?
                # If we add x, then for any a,b with a+b=x, if a,b in S, then x is in S (now).
                # So pair_sums[a] should have bit b set.
                # But we are adding x.
                # So for a=1..x-1, b=x-a.
                # If (s_mask >> b) & 1: # b was in S
                #    pair_sums[a] |= (1 << b)
                # Also symmetrically for b? pair_sums[b] |= (1 << a).
                
                for a in range(1, x):
                    b = x - a
                    if (s_mask >> b) & 1:
                        pair_sums[a] |= (1 << b)
                        pair_sums[b] |= (1 << a)
        
        size = count_bits(s_mask)
        if size > best_size:
            best_size = size
            best_mask = s_mask
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding
            for x in range(1, N+1):
                if (s_mask >> x) & 1:
                    continue
                if is_valid_addition(s_mask, x, pair_sums):
                    # Add x
                    s_mask |= (1 << x)
                    for a in range(1, x):
                        b = x - a
                        if (s_mask >> b) & 1: # b is in S (including x? No, b<x so b!=x)
                            pair_sums[a] |= (1 << b)
                            pair_sums[b] |= (1 << a)
                    size += 1
                    improved = True
                    break
            
            # 2. Try swapping (remove one, add more)
            if not improved:
                # Try removing each x and re-greedy
                # This is O(N * (N^2)) if re-greedy is slow.
                # But re-greedy with pair_sums update is fast.
                
                found_swap = False
                for x in range(1, N+1):
                    if time.time() > deadline: break
                    if not (s_mask >> x) & 1:
                        continue
                    
                    # Save state? 
                    # We can just try and revert if not better.
                    # Or copy s_mask and pair_sums.
                    # Copying pair_sums (list of 128 ints) is fast.
                    
                    old_s_mask = s_mask
                    old_pair_sums = pair_sums[:]
                    
                    # Remove x
                    s_mask &= ~(1 << x)
                    # Update pair_sums for removal of x
                    for a in range(1, x):
                        b = x - a
                        if (s_mask >> b) & 1:
                            pair_sums[a] &= ~(1 << b)
                            pair_sums[b] &= ~(1 << a)
                    
                    size_temp = size - 1
                    
                    # Greedy fill
                    filled = True
                    while filled:
                        filled = False
                        for y in range(1, N+1):
                            if (s_mask >> y) & 1:
                                continue
                            if is_valid_addition(s_mask, y, pair_sums):
                                # Add y
                                s_mask |= (1 << y)
                                for a in range(1, y):
                                    b = y - a
                                    if (s_mask >> b) & 1:
                                        pair_sums[a] |= (1 << b)
                                        pair_sums[b] |= (1 << a)
                                size_temp += 1
                                filled = True
                                break
                    
                    if size_temp > size:
                        size = size_temp
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Revert
                        s_mask = old_s_mask
                        pair_sums = old_pair_sums
                
                if not found_swap and size > 0:
                    # Random perturbation
                    bits = []
                    m = s_mask
                    while m:
                        lsb = m & -m
                        bits.append(lsb.bit_length() - 1)
                        m ^= lsb
                    if bits:
                        x_to_remove = random.choice(bits)
                        # Remove and re-greedy
                        old_s_mask = s_mask
                        old_pair_sums = pair_sums[:]
                        
                        s_mask &= ~(1 << x_to_remove)
                        for a in range(1, x_to_remove):
                            b = x_to_remove - a
                            if (s_mask >> b) & 1:
                                pair_sums[a] &= ~(1 << b)
                                pair_sums[b] &= ~(1 << a)
                        
                        size_temp = size - 1
                        filled = True
                        while filled:
                            filled = False
                            for y in range(1, N+1):
                                if (s_mask >> y) & 1:
                                    continue
                                if is_valid_addition(s_mask, y, pair_sums):
                                    s_mask |= (1 << y)
                                    for a in range(1, y):
                                        b = y - a
                                        if (s_mask >> b) & 1:
                                            pair_sums[a] |= (1 << b)
                                            pair_sums[b] |= (1 << a)
                                    size_temp += 1
                                    filled = True
                                    break
                        
                        if size_temp > best_size:
                            best_size = size_temp
                            best_mask = s_mask
                        size = size_temp
                        improved = True

        if size > best_size:
            best_size = size
            best_mask = s_mask

    result = []
    m = best_mask
    while m:
        lsb = m & -m
        result.append(lsb.bit_length() - 1)
        m ^= lsb
    
    return result
