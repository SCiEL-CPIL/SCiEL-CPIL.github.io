import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 128
    
    # Precompute masks for fast conflict checking
    # A conflict exists if a, b, c are in S and a+b, a+c, b+c are in S.
    # We can check if adding x to S creates a conflict.
    # If x is added, the new conflicts must involve x.
    # Case 1: x, a, b are the triple. Then x+a, x+b, a+b must be in S (including x if one sum is x, which is impossible for positive ints).
    # Actually, the sums a+b, a+c, b+c must be in S.
    # If the triple is (x, a, b), sums are x+a, x+b, a+b. All must be in S.
    # Note: x+a could be x? No, a>0. x+a could be a? No, x>0.
    # So x+a, x+b, a+b must be in the set S *before* adding x? 
    # Wait, if x+a is x, then a=0. Not possible.
    # If x+a is in S, and x is not in S yet, then x+a must be in S_old.
    # So we check if there exist a, b in S_old such that:
    # 1. a < b (distinct, order doesn't matter for set)
    # 2. x+a in S_old
    # 3. x+b in S_old
    # 4. a+b in S_old
    
    # Let's precompute for each potential x, the pairs (a,b) in 1..128 that would cause a conflict if x is added.
    # But S changes. So we need a dynamic check.
    
    # Bitmask approach:
    # Let S be represented as a bitmask of 128 bits.
    # For a candidate x, we want to check if there are a, b in S such that a+b, a+x, b+x are in S.
    # Let A = {a in S | a+x in S}
    # Let B = {b in S | a+b in S for some a in A? No, a+b must be in S}
    # Actually, for a fixed x, define:
    # Pairs (a,b) such that a+x in S, b+x in S, and a+b in S.
    # We can precompute for each x, a bitmask `ConflictMask[x]`? No, because S changes.
    
    # However, we can precompute `SumMask[i]` = bitmask of j such that i+j <= 128? No, that's not right.
    
    # Let's look at the structure again.
    # Forbidden: a, b, c in S AND a+b, a+c, b+c in S.
    # If we add x, the only new forbidden triples are those containing x.
    # So we need to check if there exist a, b in S_old such that:
    # {x, a, b} are distinct (a!=b, a!=x, b!=x)
    # x+a in S_old
    # x+b in S_old
    # a+b in S_old
    
    # Let's precompute for each x, the set of pairs (a,b) that would form a conflict.
    # This depends on S_old.
    
    # Efficient Check:
    # For a given x and S_old (bitmask):
    # 1. Find all a in S_old such that x+a in S_old. Let this set be A_x.
    # 2. For each a in A_x, we need to find b in S_old such that:
    #    - b != a
    #    - b != x
    #    - x+b in S_old (so b is also in A_x)
    #    - a+b in S_old
    # So we need to check if there exist a, b in A_x (a != b) such that a+b in S_old.
    # Also need to ensure a,b != x. Since a,b are in S_old and x is not in S_old, a!=x and b!=x is automatically satisfied.
    
    # So the check is:
    # 1. Compute A_x = {a | a in S_old AND (a+x) in S_old}
    # 2. Check if there exist distinct a, b in A_x such that (a+b) in S_old.
    
    # We can precompute for each x, the list of a's? No, S_old changes.
    # But we can use bit operations.
    # Let S be an integer bitmask.
    # Shift S right by x? No.
    # We want a such that bit a is set in S AND bit a+x is set in S.
    # This is `S & (S >> x)`? No, `S >> x` has bit i set if bit i+x is set in S.
    # So `S & (S >> x)` has bit i set if bit i is set in S AND bit i+x is set in S.
    # So `A_x_mask = S & (S >> x)`.
    # Now we have the mask for A_x.
    # We need to check if there are distinct a, b in A_x such that a+b is in S.
    # This is equivalent to: Does there exist a in A_x such that (A_x & (S >> a)) has a bit set other than a?
    # Or simpler: For each a in A_x, check if (A_x & (S >> a)) & ~(1<<a) is non-zero.
    
    # Iterating over set bits in A_x is O(k) where k is size of A_x.
    # A_x size is at most 64.
    # So check_add is O(k * 1) bit ops. Very fast.
    
    def can_add(x, S_mask):
        # x is 1-based index? Let's use 0-based for bits.
        # Bit i represents element i+1.
        # Element x corresponds to bit x-1.
        # Sum a+b corresponds to element a+b, bit a+b-1.
        # If a is element i+1, b is element j+1.
        # a+b is element i+j+2. Bit i+j+1.
        
        # Let's stick to 1-based values for logic, map to 0-based bits.
        # val -> bit val-1
        
        # A_x = {a in S | a+x in S}
        # a is val a. Bit a-1.
        # a+x is val a+x. Bit a+x-1.
        # Condition: bit a-1 in S AND bit a+x-1 in S.
        # Let S_mask be the bitmask.
        # We want to find a such that bit (a-1) is set in S_mask AND bit (a+x-1) is set in S_mask.
        # Shift S_mask right by x?
        # If we shift S_mask right by x, bit (a-1) in shifted mask corresponds to bit (a-1+x) in original mask.
        # So `S_mask & (S_mask >> x)` will have bit (a-1) set if bit (a-1) and bit (a-1+x) are set in S_mask.
        # This matches a and a+x.
        
        A_mask = S_mask & (S_mask >> x)
        
        # Now check if there exist distinct a, b in A_mask such that a+b in S_mask.
        # a, b are values.
        # a+b in S_mask means bit (a+b-1) is set in S_mask.
        
        # Iterate through set bits in A_mask.
        # To get set bits efficiently:
        m = A_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            a_bit = lsb.bit_length() - 1
            a_val = a_bit + 1
            
            # Check for b in A_mask, b != a, such that a+b in S_mask
            # (A_mask & (S_mask >> a_val)) will have bit (b-1) set if b in A_mask AND a+b in S_mask.
            # We need to exclude b=a.
            # So check if (A_mask & (S_mask >> a_val)) & ~lsb is non-zero.
            
            candidates = A_mask & (S_mask >> a_val)
            # Remove a itself
            candidates &= ~lsb
            
            if candidates:
                return False # Conflict found
            
            m ^= lsb
            
        return True

    def local_search_polish(S_mask, max_iters=1000):
        """Local search to try to increase size by adding elements."""
        S_mask = S_mask
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            # Try each candidate not in current
            for x in range(1, 129):
                if (S_mask >> (x-1)) & 1:
                    continue
                if can_add(x, S_mask):
                    S_mask |= (1 << (x-1))
                    improved = True
                    break
            
            if not improved:
                # Try removing one element and adding two? Or just swap?
                # Simple swap: remove u, add v.
                # If we remove u, S changes.
                # We can try to find a v that can be added to S \ {u}.
                
                # To save time, only do this if we have time.
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                for u in range(1, 129):
                    if not ((S_mask >> (u-1)) & 1):
                        continue
                    new_S = S_mask & ~(1 << (u-1))
                    # Try to add 1 or 2 elements
                    added = 0
                    for v in range(1, 129):
                        if (new_S >> (v-1)) & 1:
                            continue
                        if can_add(v, new_S):
                            new_S |= (1 << (v-1))
                            added += 1
                            if added >= 1: # Just adding 1 is a swap. Adding 2 is an improvement.
                                # We can try to add more
                                pass
                    # Actually, let's just greedily add as many as possible to new_S
                    for v in range(1, 129):
                        if (new_S >> (v-1)) & 1:
                            continue
                        if can_add(v, new_S):
                            new_S |= (1 << (v-1))
                    
                    if new_S.bit_count() > S_mask.bit_count():
                        S_mask = new_S
                        improved = True
                        break
                    elif new_S.bit_count() == S_mask.bit_count() and random.random() < 0.05:
                        # Accept equal size swap to escape local optima
                        S_mask = new_S
                        # Don't break, keep looking? Or break to restart loop?
                        # Breaking allows re-evaluation of additions.
                        break
                        
        return S_mask

    best_mask = 0
    best_size = 0
    
    # Phase 1: Randomized Greedy Construction
    # Start with empty set, add random elements if valid.
    # Or start with a random subset and polish.
    
    for trial in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initialization
        # Try different densities
        if trial % 3 == 0:
            # Start with all large numbers?
            mask = 0
            for i in range(64, 129):
                mask |= (1 << (i-1))
        elif trial % 3 == 1:
            # Start with random half
            mask = 0
            for i in range(1, 129):
                if random.random() < 0.5:
                    mask |= (1 << (i-1))
        else:
            # Start empty
            mask = 0
            
        # Greedy fill
        # Shuffle candidates
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            if (mask >> (x-1)) & 1:
                continue
            if can_add(x, mask):
                mask |= (1 << (x-1))
        
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask
            
        # Polish
        if time.time() - start_time < DEADLINE - 5:
            polished = local_search_polish(mask, max_iters=50)
            if polished.bit_count() > best_size:
                best_size = polished.bit_count()
                best_mask = polished

    # Phase 2: More aggressive local search on best
    if time.time() - start_time < DEADLINE:
        # Run longer polish on the best found
        polished = local_search_polish(best_mask, max_iters=2000)
        if polished.bit_count() > best_size:
            best_size = polished.bit_count()
            best_mask = polished

    # Convert mask to list of ints
    result = []
    for i in range(1, 129):
        if (best_mask >> (i-1)) & 1:
            result.append(i)
            
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result[:20]}...{result[-20:]}")
    
    # Verify
    S = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b:
                continue
            for c in result:
                if c <= b:
                    continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    print(f"Forbidden triple: {a},{b},{c}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
